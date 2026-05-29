import snowflake.snowpark as snowpark
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
import datetime


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CONFIG
# All tunable parameters in one place.
# Never hardcode these inside functions — change here only.
# ─────────────────────────────────────────────────────────────────────────────

# Planning months to compute SOQ for
MONTHS = ['2026-06-01']

# Stock snapshot timing — 'first' (start of month) or 'mid' (mid-month)
STOCK_DATE_TYPE = ['first']

# Increment each time the pipeline is re-run — used to version output rows
RUN_VERSION = 33

# Auto-derived today's date — used as a partition key in output tables
RUN_DATE = datetime.datetime.today().strftime('%Y%m%d')

# ABC classification → safety stock days
# A = high-revenue families (30 days stock)
# B = mid-tier families     (25 days stock)
# C = tail families         (20 days stock)
ABC = {'A': 30, 'B': 25, 'C': 20}

# Z-scores per service level
# Controls how many standard deviations of demand variability safety stock covers
# e.g. 95% service level → Z = 1.65 → covers 95% of demand spike scenarios
Z_SCORE = {95: 1.65, 90: 1.28, 85: 1.04, 80: 0.85, 99: 2.33}

# Human-readable string of ABC config — stored in output rows for auditability
STR_ABC = ', '.join([f'{k} {v}' for k, v in ABC.items()])

# OBD flag — whether this run uses OBD (On-Board Diagnostics) mapped SKUs
IS_OBD   = True
OBD_FLAG = 'Y' if IS_OBD else 'N'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Table references
# All source and destination table names centralised here.
# If a table is renamed or moved, change it once here — not inside functions.
# ─────────────────────────────────────────────────────────────────────────────

# Input: pre-computed demand forecasts + stock positions per dealer-family-SKU
BASE_SOQ_TABLE = 'MOP_DATABASE.SOQ.SOQ_BASE_TABLE_FINAL_CONCATENATED'

# Input: demand variability at SKU level (how much each SKU's sales fluctuate)
DEMAND_VARIABILITY_TABLE = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_FINAL_VERSION'

# Input: demand variability at model family level (all SKUs in a family share one value)
DEMAND_VARIABILITY_FAM_TABLE = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_MODEL_FAMILY_FINAL_VERSION'

# Output: final SOQ results — appended per run, not overwritten
SOQ_OUTPUT_TABLE = 'MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_2'

# Output: rows dropped due to missing lead time — saved for investigation
NULL_LEAD_TIME_TABLE = 'MOP_DATABASE.SOQ.NULL_LEAD_TIME'

# Input: end-of-journey (discontinuation) recommendations per SKU
END_JOURNEY_TABLE = 'MOP_DATABASE.SOQ.END_OF_JOURNEY_RECOMMENDATION'

# ─────────────────────────────────────────────────────────────────────────────
# BUG 5 FIX — DEMAND_VARIABILITY unit flag
# Set to True  if DEMAND_VARIABILITY is an absolute std deviation (same units
#              as sales volume — pieces, units, etc.)
# Set to False if DEMAND_VARIABILITY is a CV (coefficient of variation,
#              unitless ratio: std_dev / mean). In that case the formula
#              multiplies CV × PREDICTED_SALES_SKU to get units-consistent
#              standard deviation before applying Z and sqrt scaling.
# Wrong setting here silently produces incorrect safety stock — verify against
# your upstream variability table's documentation before running.
# ─────────────────────────────────────────────────────────────────────────────
DEMAND_VARIABILITY_IS_ABSOLUTE = True  # <-- verify this against your data


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — ABC helper function
# Pure Python — no Snowpark dependencies inside.
# Registered as a UDF in Step 7 so it executes server-side in Snowflake.
#
# Classifies a dealer-family based on its cumulative % of dealer's total sales:
#   cumulative < 70%  → A  (top revenue families — largest safety stock)
#   cumulative 70–90% → B  (mid-tier)
#   cumulative > 90%  → C  (tail families — smallest safety stock)
# ─────────────────────────────────────────────────────────────────────────────
def _get_abc_class(cumulative_pct: float) -> str:
    if cumulative_pct < 70:
        return 'A'
    elif cumulative_pct > 90:
        return 'C'
    else:
        return 'B'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — calculate_soq function
# One call = one SOQ scenario:
#   one planning month × one date period × one service level × one variability mode
# Results are appended to SOQ_OUTPUT_TABLE.
# main() calls this 10 times per month-period combination.
# ─────────────────────────────────────────────────────────────────────────────
def calculate_soq(
    session:                snowpark.Session,
    planning_month:         str,
    date_period:            str,
    service_level:          int  = 95,
    run_date:               str  = RUN_DATE,
    sku_demand_variability: bool = True,
    run_version:            int  = RUN_VERSION
) -> None:
    """
    Computes Suggested Order Quantity (SOQ) for one scenario.

    Parameters
    ----------
    planning_month         : First day of the planning month (YYYY-MM-DD)
    date_period            : 'first' or 'mid' — stock snapshot timing
    service_level          : Service level % — drives Z-score for safety stock
    run_date               : YYYYMMDD string — partition key in output table
    sku_demand_variability : True  → use SKU-level demand variability
                             False → use model-family-level demand variability
    run_version            : Integer version tag for this run
    """

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — Load and filter base SOQ table
    # BASE_SOQ_TABLE holds pre-computed demand forecasts and current stock
    # positions for every PARENT_DEALER_CODE + FAMILY + SKU combination.
    # Filter immediately to the relevant planning month, date period,
    # run date, version, and OBD flag — so all downstream steps work
    # on only the relevant slice of data.
    # IS_OBD is dropped after filtering — not needed further downstream.
    # ─────────────────────────────────────────────────────────────────────────
    soq_base = (
        session.table(BASE_SOQ_TABLE)
        .filter(F.col('PLANNING_MONTH')    == F.lit(planning_month))
        .filter(F.col('STOCK_DATE_PERIOD') == F.lit(date_period))
        .filter(F.col('RUN_DATE')          == F.lit(int(run_date)))  # stored as int in table
        .filter(F.col('RUN_VERSION')       == F.lit(run_version))
        .filter(F.col('IS_OBD')            == F.lit(OBD_FLAG))
        .drop('IS_OBD')
    )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — Load demand variability and join onto base data
    # Demand variability = how much a SKU's (or family's) sales fluctuate
    # month over month. Higher variability → more safety stock needed.
    #
    # Two modes controlled by sku_demand_variability:
    #   True  → variability at SKU level (granular — each SKU gets its own value)
    #   False → variability at family level (all SKUs in a family share one value)
    #
    # Left join in both cases:
    #   - Preserves all base rows even if no variability record exists
    #   - Unmatched rows get null → coalesced to 1.0 (neutral — no amplification)
    #
    # BUG 3 FIX — Duplicate column collision:
    #   The variability tables share column names with the base table
    #   (PLANNING_MONTH, RUN_DATE, RUN_VERSION). Snowpark does not
    #   deduplicate on join — duplicate names cause ambiguous reference
    #   errors downstream. Fix: drop shared columns from the right-hand
    #   table before joining, since they are redundant (already filtered on).
    # ─────────────────────────────────────────────────────────────────────────

    # Columns present in both tables that are not part of the join key —
    # must be dropped from the right side before joining to avoid duplicates.
    SHARED_NON_KEY_COLS = ['RUN_DATE', 'RUN_VERSION']

    if sku_demand_variability:

        demand_var = (
            session.table(DEMAND_VARIABILITY_TABLE)
            .filter(F.col('PLANNING_MONTH') == F.lit(planning_month))
            .filter(F.col('RUN_DATE')       == F.lit(int(run_date)))
            .filter(F.col('RUN_VERSION')    == F.lit(run_version))
            .filter(F.col('IS_OBD')         == F.lit(OBD_FLAG))
            .drop('IS_OBD')
            # BUG 3 FIX: drop columns that also exist in soq_base but are
            # not part of the join key — prevents duplicate column names post-join
            .drop(*SHARED_NON_KEY_COLS)
        )

        # SKU included in join key — each SKU gets its own variability value
        data = soq_base.join(
            demand_var,
            on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'SKU', 'PLANNING_MONTH'],
            how='left'
        )

        # Null variability → 1.0 (safety stock formula multiplies by this;
        # 1.0 means no amplification — neutral fallback)
        data = data.with_column(
            'DEMAND_VARIABILITY',
            F.coalesce(F.col('DEMAND_VARIABILITY'), F.lit(1.0))
        )

        # Tag for auditability — visible in output so analysts know which mode was used
        data = data.with_column('DEMAND_VARIABILITY_TYPE', F.lit('SKU_BASED'))

    else:

        demand_var = (
            session.table(DEMAND_VARIABILITY_FAM_TABLE)
            .filter(F.col('PLANNING_MONTH') == F.lit(planning_month))
            .filter(F.col('RUN_DATE')       == F.lit(int(run_date)))
            .filter(F.col('RUN_VERSION')    == F.lit(run_version))
            .filter(F.col('IS_OBD')         == F.lit(OBD_FLAG))
            .drop('IS_OBD')
            # BUG 3 FIX: same as above — drop shared non-key columns
            .drop(*SHARED_NON_KEY_COLS)
        )

        # SKU NOT in join key — all SKUs in the same family inherit the
        # family's variability value
        data = soq_base.join(
            demand_var,
            on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'PLANNING_MONTH'],
            how='left'
        )

        data = data.with_column(
            'DEMAND_VARIABILITY',
            F.coalesce(F.col('DEMAND_VARIABILITY'), F.lit(1.0))
        )

        data = data.with_column('DEMAND_VARIABILITY_TYPE', F.lit('MODEL_SKU_FAMILY_BASED'))


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — ABC classification
    # Classifies each FAMILY within a DEALER by its share of total dealer sales.
    # Result drives how many safety stock days each family carries (Step 8).
    #
    # Process:
    #   1. Deduplicate to dealer-family level (one row per family per dealer)
    #      Deduplication prevents double-counting when multiple SKU rows exist
    #   2. Compute each family's % share of its dealer's total predicted sales
    #   3. Sort families within each dealer by sales % descending
    #   4. Compute cumulative % — running total of % share
    #   5. Apply ABC class via UDF:
    #      cumulative < 70%  → A
    #      cumulative 70–90% → B
    #      cumulative > 90%  → C
    #   6. Join ABC class back onto main dataset
    # ─────────────────────────────────────────────────────────────────────────

    # Window over the entire dealer — used for dealer-level sales total
    dealer_window = Window.partition_by('PARENT_DEALER_CODE')

    # Window over dealer, sorted by sales descending — used for cumulative %
    # Sorted descending so highest-revenue families accumulate first → land in A
    family_window = (
        Window.partition_by('PARENT_DEALER_CODE')
              .order_by(F.col('PREDICTED_SALES').desc())
    )

    abc_data = (
        data
        # One row per dealer-family — prevents SKU rows from inflating totals
        .select('PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'PREDICTED_SALES')
        .distinct()

        # Dealer total predicted sales — denominator for % share calculation
        .with_column(
            'DEALER_PREDICTED_SALES',
            F.sum('PREDICTED_SALES').over(dealer_window)
        )

        # Each family's % contribution to its dealer's total predicted sales
        # Guard against division by zero — assign 0% if dealer has no sales
        .with_column(
            'PERC_SALES',
            F.when(
                F.col('DEALER_PREDICTED_SALES') == F.lit(0), F.lit(0.0)
            ).otherwise(
                (F.col('PREDICTED_SALES') / F.col('DEALER_PREDICTED_SALES')) * F.lit(100)
            )
        )

        # Cumulative % — running sum of % share within dealer, sorted by sales desc
        # Families are sorted highest to lowest so A-class families accumulate first
        .with_column(
            'CUMULATIVE_PERCENT_SALES',
            F.sum('PERC_SALES').over(
                family_window.rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
        )
    )

    # Register _get_abc_class as a UDF — runs server-side in Snowflake.
    # UDF is necessary because the conditional logic (if/elif/else) cannot
    # be expressed as a native Snowpark expression.
    # replace=True: re-registers on every run — safe for iterative development.
    get_abc_udf = session.udf.register(
        _get_abc_class,
        name='get_abc_class_udf',
        replace=True,
        input_types=[snowpark.types.FloatType()],
        return_type=snowpark.types.StringType()
    )

    abc_data = (
        abc_data
        # Apply UDF — assigns A/B/C to each dealer-family row
        .with_column('ABC', get_abc_udf(F.col('CUMULATIVE_PERCENT_SALES')))
        # Keep only columns needed for the join back onto main dataset
        .select('PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'ABC')
    )

    # Join ABC class onto main dataset at dealer-family level
    # Left join preserves all rows; unclassified families get null ABC → handled in Step 8
    data = data.join(abc_data, on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE'], how='left')


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8 — Map ABC class to safety stock days
    # A-class families get 30 days — stockouts on high-revenue families are costly
    # C-class families get 20 days — tail families have lower business impact
    # Default 15 days — fallback for any row where ABC classification failed
    # ─────────────────────────────────────────────────────────────────────────
    data = data.with_column(
        'SAFETY_STOCK_DAYS',
        F.when(F.col('ABC') == F.lit('A'), F.lit(ABC['A']))
         .when(F.col('ABC') == F.lit('B'), F.lit(ABC['B']))
         .when(F.col('ABC') == F.lit('C'), F.lit(ABC['C']))
         .otherwise(F.lit(15))
    )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9 — Handle null lead time rows
    # MAX_LEAD_TIME is required for SOQ computation — without it lead time
    # stock cannot be calculated and the SOQ would be incomplete.
    # Rows with null lead time are saved separately for investigation,
    # then dropped from the main pipeline.
    #
    # BUG 2 FIX — Null lead time table always overwrites:
    #   calculate_soq is called 10 times per month-period (5 service levels
    #   × 2 variability modes). Overwrite mode means each call wipes the
    #   previous call's null rows — a later call with zero nulls clears the
    #   table entirely.
    #   Fix: switch to append mode and tag each row with SERVICE_LEVEL and
    #   DEMAND_VARIABILITY_TYPE so you can filter by specific run when investigating.
    # ─────────────────────────────────────────────────────────────────────────

    # Separate rows with null MAX_LEAD_TIME
    null_lead_time_rows = data.filter(F.col('MAX_LEAD_TIME').is_null())
    null_count          = null_lead_time_rows.count()

    if null_count > 0:
        # BUG 2 FIX: tag rows with run context before appending
        null_lead_time_rows = (
            null_lead_time_rows
            .with_column('SERVICE_LEVEL',          F.lit(service_level))
            .with_column('DEMAND_VARIABILITY_TYPE', F.col('DEMAND_VARIABILITY_TYPE'))
            .with_column('CAPTURED_AT',            F.lit(run_date))
        )
        # BUG 2 FIX: append instead of overwrite — preserves rows from all 10 calls
        null_lead_time_rows.write.mode('append').save_as_table(NULL_LEAD_TIME_TABLE)

    # Continue with only rows that have a valid lead time
    data = data.filter(F.col('MAX_LEAD_TIME').is_not_null())


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 10 — Lead time stock
    # How much stock is consumed while waiting for a replenishment order to arrive.
    #
    # Formula: (PREDICTED_SALES_SKU × LEAD_TIME_DAYS) / 30
    #   PREDICTED_SALES_SKU : monthly predicted sales for this SKU
    #   LEAD_TIME_DAYS      : how many days the supplier takes to deliver
    #   / 30                : converts monthly sales to daily rate,
    #                         then multiplies by lead time days
    #   F.ceil              : always round up — never under-stock due to rounding
    #
    # Three variants: MAX, MIN, AVG lead time
    #   MAX = worst case (longest supplier lead time — most conservative)
    #   MIN = best case  (shortest lead time — most aggressive)
    #   AVG = balanced   (average historical lead time)
    # ─────────────────────────────────────────────────────────────────────────
    data = (
        data
        .with_column(
            'MAX_LEAD_TIME_STOCK',
            F.ceil((F.col('PREDICTED_SALES_SKU') * F.col('MAX_LEAD_TIME')) / F.lit(30))
        )
        .with_column(
            'MIN_LEAD_TIME_STOCK',
            F.ceil((F.col('PREDICTED_SALES_SKU') * F.col('MIN_LEAD_TIME')) / F.lit(30))
        )
        .with_column(
            'AVG_LEAD_TIME_STOCK',
            F.ceil((F.col('PREDICTED_SALES_SKU') * F.col('AVG_LEAD_TIME')) / F.lit(30))
        )
    )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 11 — Safety stock
    # Buffer stock that protects against demand variability during the
    # safety stock coverage period.
    #
    # Formula: DEMAND_VARIABILITY × sqrt(SAFETY_STOCK_DAYS / 30) × Z_SCORE
    #
    #   DEMAND_VARIABILITY     : std deviation of demand — higher = more volatile SKU
    #   sqrt(SS_DAYS / 30)     : scales variability to the safety stock horizon.
    #                            Square root is used because demand variance compounds
    #                            sub-linearly over independent time periods
    #                            (statistical property — not an arbitrary choice)
    #   Z_SCORE                : from service level config — 95% → 1.65, meaning
    #                            safety stock covers 95% of demand spike scenarios
    #   F.ceil                 : always round up
    #
    # BUG 5 FIX — DEMAND_VARIABILITY unit assumption:
    #   If DEMAND_VARIABILITY is a CV (std_dev / mean, unitless), multiplying
    #   it directly by sqrt and Z gives a result in the wrong units — the cap
    #   of 3 × PREDICTED_SALES_SKU (in pieces) is then comparing apples to
    #   oranges, and safety stock will be massively overstated for volatile
    #   low-volume SKUs.
    #   Fix: when DEMAND_VARIABILITY_IS_ABSOLUTE = False, convert CV back to
    #   absolute std deviation first by multiplying by PREDICTED_SALES_SKU
    #   before applying the Z and sqrt scaling.
    #   Verify DEMAND_VARIABILITY_IS_ABSOLUTE against your upstream table docs.
    #
    # Cap at 3× monthly predicted sales:
    #   Prevents extreme safety stock on very volatile but low-volume SKUs
    #   where the formula can produce unrealistically large numbers.
    #
    # Coalesce to 0 after cap:
    #   Safe fallback in case PREDICTED_SALES_SKU was null (cap produces null)
    #
    # STK_AS_ON_DATE null → 0:
    #   Null stock record means the dealer has no stock — treat as zero
    # ─────────────────────────────────────────────────────────────────────────

    # Resolve Z-score for this service level from config
    z = Z_SCORE[service_level]

    if DEMAND_VARIABILITY_IS_ABSOLUTE:
        # DEMAND_VARIABILITY is already in units of sales (pieces, units, etc.)
        # Use directly in the formula — no conversion needed
        raw_safety_stock = (
            F.col('DEMAND_VARIABILITY')
            * F.sqrt(F.col('SAFETY_STOCK_DAYS') / F.lit(30))
            * F.lit(z)
        )
    else:
        # DEMAND_VARIABILITY is a CV (unitless ratio: std_dev / mean)
        # Convert to absolute std deviation first: CV × mean (PREDICTED_SALES_SKU)
        # Then apply Z and sqrt scaling — result is now in units of sales
        raw_safety_stock = (
            F.col('DEMAND_VARIABILITY')
            * F.col('PREDICTED_SALES_SKU')
            * F.sqrt(F.col('SAFETY_STOCK_DAYS') / F.lit(30))
            * F.lit(z)
        )

    data = data.with_column(
        'SAFETY_STOCK',
        F.least(
            # Uncapped safety stock
            F.ceil(raw_safety_stock),
            # Cap: safety stock cannot exceed 3× monthly predicted SKU sales
            F.col('PREDICTED_SALES_SKU') * F.lit(3)
        )
    )

    # Null fallback after cap
    data = data.with_column(
        'SAFETY_STOCK',
        F.coalesce(F.col('SAFETY_STOCK'), F.lit(0))
    )

    # Null stock on date → zero (no stock record = no stock at this dealer)
    data = data.with_column(
        'STK_AS_ON_DATE',
        F.coalesce(F.col('STK_AS_ON_DATE'), F.lit(0))
    )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 12 — SOQ Approach 1: full month coverage
    # Suggests enough stock to cover:
    #   predicted demand + safety stock buffer + lead time stock
    # Then subtracts existing stock on hand — order only what's actually needed.
    #
    # REORDER_STOCK   = safety stock + lead time stock  (the buffer)
    # TOTAL_STOCK_SKU = predicted demand + reorder stock (total needed)
    # Suggested_Stock = total needed - stock on hand    (what to order)
    # Adjusted_Order  = max(suggested, 0)               (never negative)
    #
    # Three scenarios: MAX / MIN / AVG lead time — same formula, different inputs
    # ─────────────────────────────────────────────────────────────────────────
    data = (
        data

        # MAX lead time scenario — most conservative
        .with_column('MAX_REORDER_STOCK',
                     F.col('SAFETY_STOCK') + F.col('MAX_LEAD_TIME_STOCK'))
        .with_column('MAX_TOTAL_STOCK_SKU',
                     F.col('PREDICTED_SALES_SKU') + F.col('MAX_REORDER_STOCK'))
        .with_column('MAX_Suggested_Stock_SKU',
                     F.col('MAX_TOTAL_STOCK_SKU') - F.col('STK_AS_ON_DATE'))
        .with_column('MAX_Adjusted_Monthly_Order',
                     F.greatest(F.col('MAX_Suggested_Stock_SKU'), F.lit(0)))

        # MIN lead time scenario — most aggressive
        .with_column('MIN_REORDER_STOCK',
                     F.col('SAFETY_STOCK') + F.col('MIN_LEAD_TIME_STOCK'))
        .with_column('MIN_TOTAL_STOCK_SKU',
                     F.col('PREDICTED_SALES_SKU') + F.col('MIN_REORDER_STOCK'))
        .with_column('MIN_Suggested_Stock_SKU',
                     F.col('MIN_TOTAL_STOCK_SKU') - F.col('STK_AS_ON_DATE'))
        .with_column('MIN_Adjusted_Monthly_Order',
                     F.greatest(F.col('MIN_Suggested_Stock_SKU'), F.lit(0)))

        # AVG lead time scenario — balanced
        .with_column('AVG_REORDER_STOCK',
                     F.col('SAFETY_STOCK') + F.col('AVG_LEAD_TIME_STOCK'))
        .with_column('AVG_TOTAL_STOCK_SKU',
                     F.col('PREDICTED_SALES_SKU') + F.col('AVG_REORDER_STOCK'))
        .with_column('AVG_Suggested_Stock_SKU',
                     F.col('AVG_TOTAL_STOCK_SKU') - F.col('STK_AS_ON_DATE'))
        .with_column('AVG_Adjusted_Monthly_Order',
                     F.greatest(F.col('AVG_Suggested_Stock_SKU'), F.lit(0)))
    )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 13 — SOQ Approach 2: reorder point only
    # A leaner alternative to Approach 1.
    # Suggests only enough to replenish to the reorder point
    # (safety stock + lead time stock) — does NOT include the full month's demand.
    # Appropriate when the dealer already has sufficient forward stock.
    #
    # Formula: REORDER_STOCK - STK_AS_ON_DATE (floored at 0)
    # ─────────────────────────────────────────────────────────────────────────
    data = (
        data
        .with_column('AVG_SOQ_APPROACH_2',
                     F.col('AVG_REORDER_STOCK') - F.col('STK_AS_ON_DATE'))
        .with_column('MAX_SOQ_APPROACH_2',
                     F.col('MAX_REORDER_STOCK') - F.col('STK_AS_ON_DATE'))
        .with_column('MIN_SOQ_APPROACH_2',
                     F.col('MIN_REORDER_STOCK') - F.col('STK_AS_ON_DATE'))
        .with_column('AVG_Adjusted_Monthly_Order_APPROACH_2',
                     F.greatest(F.col('AVG_SOQ_APPROACH_2'), F.lit(0)))
        .with_column('MAX_Adjusted_Monthly_Order_APPROACH_2',
                     F.greatest(F.col('MAX_SOQ_APPROACH_2'), F.lit(0)))
        .with_column('MIN_Adjusted_Monthly_Order_APPROACH_2',
                     F.greatest(F.col('MIN_SOQ_APPROACH_2'), F.lit(0)))
    )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 14 — Audit columns
    # Tag every output row with the exact run configuration that produced it.
    # Done last — after all business logic — so these columns don't interfere
    # with any joins or calculations above.
    # Allows any output row to be fully traced and reproduced.
    #
    # BUG 1 FIX — ABC column overwrite:
    #   The original code overwrote the per-row ABC classification ('A'/'B'/'C')
    #   computed in Step 7 with the audit config string ("A 30, B 25, C 20").
    #   Step 8's safety stock math ran before this so the numbers were correct,
    #   but the output ABC column was useless for filtering or reporting.
    #   Fix: store the audit string under a separate column ABC_CONFIG.
    #   ABC in the output now correctly holds the per-row classification.
    # ─────────────────────────────────────────────────────────────────────────
    data = (
        data
        # BUG 1 FIX: renamed from 'ABC' to 'ABC_CONFIG' — preserves per-row ABC
        .with_column('ABC_CONFIG',    F.lit(STR_ABC))       # ABC thresholds used in this run
        .with_column('Z_SCORE',       F.lit(z))             # Z-score for this service level
        .with_column('SERVICE_LEVEL', F.lit(service_level)) # service level %
        .with_column('RUN_DATE',      F.lit(run_date))      # date this run was executed
        .with_column('RUN_VERSION',   F.lit(run_version))   # version tag
        .with_column('IS_OBD',        F.lit(OBD_FLAG))      # OBD flag for this run
    )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 15 — End of journey join
    # Some SKUs have a discontinuation recommendation (end-of-journey flag).
    # Join the latest recommendation for each SKU onto the SOQ output.
    # "Latest" = max RUN_DATE in END_JOURNEY_TABLE — ensures freshest data.
    # Left join: SOQ rows with no end-of-journey record get a null flag.
    # This is enrichment only — not used in any SOQ calculation above.
    # Done last so it doesn't interfere with any business logic.
    #
    # BUG 4 FIX — EOJ join fan-out:
    #   If the same SKU appears more than once in END_JOURNEY_TABLE on the
    #   max run date (e.g. different dealers or families), joining on SKU
    #   alone fans out — one input row produces multiple output rows,
    #   silently inflating the dataset.
    #   Fix: deduplicate END_JOURNEY_TABLE to one row per SKU before joining.
    #   If multiple recommendations exist for the same SKU, take the most
    #   conservative one (RECOMMEND_END_OF_JOURNEY = 'Y' takes priority).
    # ─────────────────────────────────────────────────────────────────────────

    # Collect the max run date from the end-of-journey table
    # Single-row aggregation — cheap eager action
    max_eoj_run_date = (
        session.table(END_JOURNEY_TABLE)
               .select(F.max('RUN_DATE'))
               .collect()[0][0]
    )

    # BUG 4 FIX: deduplicate to one row per SKU before joining.
    # Priority: if any record for a SKU recommends end-of-journey ('Y'),
    # that recommendation wins — most conservative outcome.
    end_journey = (
        session.table(END_JOURNEY_TABLE)
        .filter(F.col('RUN_DATE') == F.lit(max_eoj_run_date))
        .select('SKU', 'RECOMMEND_END_OF_JOURNEY')
        # Group by SKU — take 'Y' if any record recommends end-of-journey,
        # otherwise take whatever value exists (max of 'Y'/'N' = 'Y')
        .group_by('SKU')
        .agg(F.max('RECOMMEND_END_OF_JOURNEY').alias('RECOMMEND_END_OF_JOURNEY'))
    )

    # Left join on SKU — unmatched SKUs get null RECOMMEND_END_OF_JOURNEY
    data = data.join(end_journey, on='SKU', how='left')


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 16 — Write output
    # Append mode — NOT overwrite.
    # main() calls calculate_soq 10 times per month-period combination
    # (5 service levels × 2 variability modes). All 10 runs must coexist
    # in the same output table — overwrite would erase previous runs.
    # ─────────────────────────────────────────────────────────────────────────
    data.write.mode('append').save_as_table(SOQ_OUTPUT_TABLE)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 17 — main: orchestrator
# Loops over all combinations of months, date periods, service levels,
# and variability modes — calling calculate_soq once per combination.
#
# Total runs per month × period = 10:
#   5 service levels (80, 85, 90, 95, 99) × 2 variability modes (SKU, family)
# ─────────────────────────────────────────────────────────────────────────────
def main(session: snowpark.Session):
    """
    Entry point for the SOQ pipeline.
    Orchestrates all SOQ scenarios and returns the output table.
    """
    for month in MONTHS:
        for period in STOCK_DATE_TYPE:
            for service_level in Z_SCORE.keys():

                # Run 1 of 2: SKU-level demand variability
                # More granular — each SKU gets its own variability value
                calculate_soq(
                    session, month, period,
                    service_level=service_level,
                    run_date=RUN_DATE,
                    sku_demand_variability=True,
                    run_version=RUN_VERSION
                )

                # Run 2 of 2: family-level demand variability
                # All SKUs in the same family share one variability value
                calculate_soq(
                    session, month, period,
                    service_level=service_level,
                    run_date=RUN_DATE,
                    sku_demand_variability=False,
                    run_version=RUN_VERSION
                )

    return session.table(SOQ_OUTPUT_TABLE)