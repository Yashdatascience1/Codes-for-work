import snowflake.snowpark as snowpark
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
import math
import logging
import datetime

# ── Logging setup ─────────────────────────────────────────────────────────────
# Configure logger at module level so it's available across all functions.
# Level INFO captures operational events; use DEBUG for deeper tracing.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Planning months for which SOQ needs to be computed.
MONTHS = ['2026-06-01']

# Whether to compute SOQ at first or mid of month stock date.
STOCK_DATE_TYPE = ['first']

# Incrementing run version — used to version-control each SOQ computation run.
RUN_VERSION = 33

# ABC classification thresholds (cumulative % of predicted sales within a dealer).
# A = top revenue families (cumulative % < 70), B = mid, C = tail (cumulative % > 90).
ABC = {'A': 30, 'B': 25, 'C': 20}   # maps ABC class → safety stock days

# Auto-derive today's run date; used as a partition key in output tables.
RUN_DATE = datetime.datetime.today().strftime('%Y%m%d')

# Z-scores for supported service levels.
# Z-score determines how many standard deviations of demand variability
# the safety stock should cover.
Z_SCORE = {95: 1.65, 90: 1.28, 80: 0.85, 85: 1.04, 99: 2.33}

# Human-readable string of ABC config stored in output for auditability.
STR_ABC = ', '.join([f'{k} {v}' for k, v in ABC.items()])

# OBD flag — whether this run uses OBD (On-Board Diagnostics) mapped SKUs.
IS_OBD = True
OBD_FLAG = 'Y' if IS_OBD else 'N'

# ── Source & destination tables ───────────────────────────────────────────────
BASE_SOQ_TABLE              = 'MOP_DATABASE.SOQ.SOQ_BASE_TABLE_FINAL_CONCATENATED'
DEMAND_VARIABILITY_TABLE    = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_FINAL_VERSION'
DEMAND_VARIABILITY_FAM_TABLE= 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_MODEL_FAMILY_FINAL_VERSION'
SOQ_OUTPUT_TABLE            = 'MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_2'
NULL_LEAD_TIME_TABLE        = 'MOP_DATABASE.SOQ.NULL_LEAD_TIME'
END_JOURNEY_TABLE           = 'MOP_DATABASE.SOQ.END_OF_JOURNEY_RECOMMENDATION'


# ── ABC classification helper ─────────────────────────────────────────────────
def _get_abc_class(cumulative_pct: float) -> str:
    """
    Assign ABC class based on cumulative % of predicted sales within a dealer.
    A = high-value families (top 70% of dealer sales).
    B = mid-value families (70–90%).
    C = tail families (beyond 90%).
    """
    if cumulative_pct < 70:
        return 'A'
    elif cumulative_pct > 90:
        return 'C'
    else:
        return 'B'


# ── Core SOQ computation ──────────────────────────────────────────────────────
def calculate_soq(
    session:              snowpark.Session,
    planning_month:       str,
    date_period:          str,
    service_level:        int  = 95,
    run_date:             str  = RUN_DATE,
    sku_demand_variability: bool = True,
    run_version:          int  = RUN_VERSION
) -> None:
    """
    Compute Suggested Order Quantity (SOQ) for a given planning month,
    stock date period, and service level. Results are appended to SOQ_OUTPUT_TABLE.

    Parameters
    ----------
    planning_month        : First day of the month being planned (YYYY-MM-DD).
    date_period           : 'first' or 'mid' — stock snapshot timing.
    service_level         : Service level % — drives Z-score for safety stock.
    run_date              : Date string (YYYYMMDD) used for table partitioning.
    sku_demand_variability: True → use SKU-level demand variability;
                            False → use model-family-level demand variability.
    run_version           : Integer version tag for this run.
    """

    logger.info(
        f"Starting SOQ calculation | month={planning_month} | period={date_period} "
        f"| service_level={service_level}% | sku_variability={sku_demand_variability} "
        f"| run_version={run_version}"
    )

    # ── Step 1: Load base SOQ data ────────────────────────────────────────────
    # BASE_SOQ_TABLE contains pre-computed demand forecasts and stock positions
    # for each PARENT_DEALER_CODE + FAMILY + SKU combination.
    # Filter to the specific planning month, date period, run date, version, and OBD flag.
    logger.info("Loading base SOQ table")

    soq_base = (
        session.table(BASE_SOQ_TABLE)
        .filter(F.col('PLANNING_MONTH')   == F.lit(planning_month))
        .filter(F.col('STOCK_DATE_PERIOD')== F.lit(date_period))
        .filter(F.col('RUN_DATE')         == F.lit(int(run_date)))
        .filter(F.col('RUN_VERSION')      == F.lit(run_version))
        .filter(F.col('IS_OBD')           == F.lit(OBD_FLAG))
        .drop('IS_OBD')   # drop after filtering; not needed downstream
    )

    logger.info(f"Base SOQ rows loaded: {soq_base.count()}")

    # ── Step 2: Load demand variability ──────────────────────────────────────
    # Demand variability measures how much a SKU's (or family's) sales fluctuate
    # month over month. Higher variability → higher safety stock required.
    # Two modes:
    #   sku_demand_variability=True  → variability computed per SKU
    #   sku_demand_variability=False → variability computed per model family
    # If a SKU/family has no variability record, default to 1 (neutral multiplier).

    if sku_demand_variability:
        logger.info("Loading SKU-level demand variability")

        demand_var = (
            session.table(DEMAND_VARIABILITY_TABLE)
            .filter(F.col('PLANNING_MONTH') == F.lit(planning_month))
            .filter(F.col('RUN_DATE')       == F.lit(int(run_date)))
            .filter(F.col('RUN_VERSION')    == F.lit(run_version))
            .filter(F.col('IS_OBD')         == F.lit(OBD_FLAG))
            .drop('IS_OBD')
        )

        # Join SKU-level variability onto base data.
        # Left join preserves all base rows; unmatched SKUs get null variability.
        data = soq_base.join(
            demand_var,
            on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'SKU', 'PLANNING_MONTH'],
            how='left'
        )

        # Fill null variability with 1 — neutral, no amplification of safety stock.
        data = data.with_column(
            'DEMAND_VARIABILITY',
            F.coalesce(F.col('DEMAND_VARIABILITY'), F.lit(1.0))
        )

        # Tag variability type for auditability in output.
        data = data.with_column('DEMAND_VARIABILITY_TYPE', F.lit('SKU_BASED'))

    else:
        logger.info("Loading model-family-level demand variability")

        demand_var = (
            session.table(DEMAND_VARIABILITY_FAM_TABLE)
            .filter(F.col('PLANNING_MONTH') == F.lit(planning_month))
            .filter(F.col('RUN_DATE')       == F.lit(int(run_date)))
            .filter(F.col('RUN_VERSION')    == F.lit(run_version))
            .filter(F.col('IS_OBD')         == F.lit(OBD_FLAG))
            .drop('IS_OBD')
        )

        # Join family-level variability — note SKU is NOT in the join key here,
        # so all SKUs within a family share the same variability value.
        data = soq_base.join(
            demand_var,
            on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'PLANNING_MONTH'],
            how='left'
        )

        data = data.with_column(
            'DEMAND_VARIABILITY',
            F.coalesce(F.col('DEMAND_VARIABILITY'), F.lit(1.0))
        )

        data = data.with_column(
            'DEMAND_VARIABILITY_TYPE', F.lit('MODEL_SKU_FAMILY_BASED')
        )

    logger.info(f"Demand variability joined | rows: {data.count()}")

    # ── Step 3: ABC classification ────────────────────────────────────────────
    # ABC classifies each FAMILY within a DEALER by its share of total predicted sales.
    # Purpose: higher-value families (A) get more safety stock days than tail families (C).
    #
    # Process:
    #   1. Compute each family's % share of dealer's total predicted sales.
    #   2. Sort families within each dealer by % share descending.
    #   3. Compute cumulative % — running total of % share.
    #   4. Assign ABC class from cumulative %:
    #      cumulative < 70%  → A (top revenue families)
    #      cumulative 70–90% → B (mid-tier)
    #      cumulative > 90%  → C (tail families)

    logger.info("Computing ABC classification")

    # Dealer-level total predicted sales — denominator for % share.
    dealer_window = Window.partition_by('PARENT_DEALER_CODE')
    dealer_total  = F.sum('PREDICTED_SALES').over(dealer_window)

    # Family-level % share within dealer.
    # Deduplicate to PARENT_DEALER_CODE + FAMILY level before computing shares —
    # avoids inflating totals if multiple SKU rows exist per family.
    family_window = Window.partition_by('PARENT_DEALER_CODE').order_by(
        F.col('PREDICTED_SALES').desc()
    )

    abc_data = (
        data.select('PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'PREDICTED_SALES')
        .distinct()   # one row per dealer-family; prevents double-counting SKUs
        .with_column('DEALER_PREDICTED_SALES', dealer_total)
        # % contribution of this family to its dealer's total sales.
        .with_column(
            'PERC_SALES',
            F.when(
                F.col('DEALER_PREDICTED_SALES') == F.lit(0), F.lit(0.0)
            ).otherwise(
                (F.col('PREDICTED_SALES') / F.col('DEALER_PREDICTED_SALES')) * F.lit(100)
            )
        )
        # Running cumulative % — sorted descending so highest-value families
        # accumulate first, correctly placing them in category A.
        .with_column(
            'CUMULATIVE_PERCENT_SALES',
            F.sum('PERC_SALES').over(family_window.rowsBetween(
                Window.unboundedPreceding, Window.currentRow
            ))
        )
    )

    # Apply ABC classification using a UDF wrapping the helper function.
    # UDF is needed because getABC contains Python conditional logic
    # that cannot be expressed as a native Snowpark expression.
    get_abc_udf = session.udf.register(
        _get_abc_class,
        name='get_abc_class_udf',
        replace=True,
        input_types=[snowpark.types.FloatType()],
        return_type=snowpark.types.StringType()
    )

    abc_data = (
        abc_data
        .with_column('ABC', get_abc_udf(F.col('CUMULATIVE_PERCENT_SALES')))
        .select('PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'ABC')
    )

    # Join ABC class back onto the main dataset.
    data = data.join(abc_data, on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE'], how='left')

    logger.info("ABC classification joined")

    # ── Step 4: Safety stock days from ABC class ──────────────────────────────
    # Map ABC class to number of safety stock days.
    # A-class families carry more safety stock (30 days) than C-class (20 days)
    # because stockouts on high-value families have larger business impact.
    # Default 15 days for any family that couldn't be classified.
    data = data.with_column(
        'SAFETY_STOCK_DAYS',
        F.when(F.col('ABC') == F.lit('A'), F.lit(ABC['A']))
         .when(F.col('ABC') == F.lit('B'), F.lit(ABC['B']))
         .when(F.col('ABC') == F.lit('C'), F.lit(ABC['C']))
         .otherwise(F.lit(15))   # fallback for unclassified families
    )

    # ── Step 5: Isolate and persist null lead time rows ───────────────────────
    # Rows with null MAX_LEAD_TIME cannot have lead time stock computed.
    # Save them to a separate table for investigation before dropping.
    logger.info("Checking for null lead time rows")

    null_lead_time_rows = data.filter(F.col('MAX_LEAD_TIME').is_null())
    null_count = null_lead_time_rows.count()

    if null_count > 0:
        logger.warning(
            f"{null_count} rows have null MAX_LEAD_TIME — "
            f"saving to {NULL_LEAD_TIME_TABLE} and excluding from SOQ calculation"
        )
        null_lead_time_rows.write.mode('overwrite').save_as_table(NULL_LEAD_TIME_TABLE)
    else:
        logger.info("No null lead time rows found")

    # Drop null lead time rows — SOQ cannot be computed without lead time.
    data = data.filter(F.col('MAX_LEAD_TIME').is_not_null())

    logger.info(f"Rows after removing null lead time: {data.count()}")

    # ── Step 6: Lead time stock ───────────────────────────────────────────────
    # Lead time stock = how much stock is consumed during the replenishment lead time.
    # Formula: (PREDICTED_SALES_SKU × LEAD_TIME_DAYS) / 30
    # Divide by 30 to convert monthly sales to daily rate, then multiply by lead time days.
    # Ceiling applied — always round up to avoid under-stocking.
    #
    # Three variants computed: MAX, MIN, AVG lead time — used later for SOQ scenarios.
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

    # ── Step 7: Safety stock ──────────────────────────────────────────────────
    # Safety stock protects against demand variability during the safety stock period.
    #
    # Formula: DEMAND_VARIABILITY × sqrt(SAFETY_STOCK_DAYS / 30) × Z_SCORE
    #
    # - DEMAND_VARIABILITY: standard deviation of demand (higher = more volatile SKU).
    # - sqrt(SAFETY_STOCK_DAYS / 30): scales variability to the safety stock horizon.
    #   Uses square root because variability compounds sub-linearly over time
    #   (statistical property of independent demand periods).
    # - Z_SCORE: driven by service level — 95% service level → Z = 1.65,
    #   meaning safety stock covers 95% of demand variability scenarios.
    # Ceiling applied — always round up.
    #
    # Cap: safety stock cannot exceed 3× predicted monthly sales.
    # Prevents pathologically large safety stocks for very volatile low-volume SKUs.
    z = Z_SCORE[service_level]

    data = data.with_column(
        'SAFETY_STOCK',
        F.least(
            # Uncapped safety stock formula
            F.ceil(
                F.col('DEMAND_VARIABILITY')
                * F.sqrt(F.col('SAFETY_STOCK_DAYS') / F.lit(30))
                * F.lit(z)
            ),
            # Cap at 3× monthly predicted sales
            F.col('PREDICTED_SALES_SKU') * F.lit(3)
        )
    )

    # Replace any remaining null safety stock with 0 — safe fallback.
    data = data.with_column(
        'SAFETY_STOCK',
        F.coalesce(F.col('SAFETY_STOCK'), F.lit(0))
    )

    # ── Step 8: Fill null stock on date ───────────────────────────────────────
    # STK_AS_ON_DATE = current physical stock at the dealer as of the stock date.
    # Null means no stock record exists — treat as zero stock.
    data = data.with_column(
        'STK_AS_ON_DATE',
        F.coalesce(F.col('STK_AS_ON_DATE'), F.lit(0))
    )

    # ── Step 9: SOQ Approach 1 — full month coverage ──────────────────────────
    # Approach 1 suggests enough stock to cover:
    #   predicted sales + safety stock + lead time stock
    # then subtracts existing stock on hand.
    # Three scenarios: MAX / MIN / AVG lead time.
    #
    # MAX scenario: most conservative — assumes worst-case (longest) lead time.
    # MIN scenario: most aggressive — assumes best-case (shortest) lead time.
    # AVG scenario: balanced — uses average lead time.
    #
    # Adjusted Monthly Order floors at 0 — never suggest negative orders.
    data = (
        data
        # MAX scenario
        .with_column('MAX_REORDER_STOCK',
                     F.col('SAFETY_STOCK') + F.col('MAX_LEAD_TIME_STOCK'))
        .with_column('MAX_TOTAL_STOCK_SKU',
                     F.col('PREDICTED_SALES_SKU') + F.col('MAX_REORDER_STOCK'))
        .with_column('MAX_Suggested_Stock_SKU',
                     F.col('MAX_TOTAL_STOCK_SKU') - F.col('STK_AS_ON_DATE'))
        .with_column('MAX_Adjusted_Monthly_Order',
                     F.greatest(F.col('MAX_Suggested_Stock_SKU'), F.lit(0)))

        # MIN scenario
        .with_column('MIN_REORDER_STOCK',
                     F.col('SAFETY_STOCK') + F.col('MIN_LEAD_TIME_STOCK'))
        .with_column('MIN_TOTAL_STOCK_SKU',
                     F.col('PREDICTED_SALES_SKU') + F.col('MIN_REORDER_STOCK'))
        .with_column('MIN_Suggested_Stock_SKU',
                     F.col('MIN_TOTAL_STOCK_SKU') - F.col('STK_AS_ON_DATE'))
        .with_column('MIN_Adjusted_Monthly_Order',
                     F.greatest(F.col('MIN_Suggested_Stock_SKU'), F.lit(0)))

        # AVG scenario
        .with_column('AVG_REORDER_STOCK',
                     F.col('SAFETY_STOCK') + F.col('AVG_LEAD_TIME_STOCK'))
        .with_column('AVG_TOTAL_STOCK_SKU',
                     F.col('PREDICTED_SALES_SKU') + F.col('AVG_REORDER_STOCK'))
        .with_column('AVG_Suggested_Stock_SKU',
                     F.col('AVG_TOTAL_STOCK_SKU') - F.col('STK_AS_ON_DATE'))
        .with_column('AVG_Adjusted_Monthly_Order',
                     F.greatest(F.col('AVG_Suggested_Stock_SKU'), F.lit(0)))
    )

    # ── Step 10: SOQ Approach 2 — reorder point only ─────────────────────────
    # Approach 2 is leaner: suggests only enough to replenish to the reorder point
    # (safety stock + lead time stock), NOT the full month's predicted demand.
    # This is appropriate when the dealer already has sufficient forward stock.
    #
    # Formula: REORDER_STOCK - STK_AS_ON_DATE (floored at 0).
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

    # ── Step 11: Audit columns ────────────────────────────────────────────────
    # Tag every output row with run metadata for traceability and reproducibility.
    data = (
        data
        .with_column('ABC',           F.lit(STR_ABC))        # ABC config used in this run
        .with_column('Z_SCORE',       F.lit(z))              # Z-score used
        .with_column('SERVICE_LEVEL', F.lit(service_level))  # service level %
        .with_column('RUN_DATE',      F.lit(run_date))       # date of this run
        .with_column('RUN_VERSION',   F.lit(run_version))    # version of this run
        .with_column('IS_OBD',        F.lit(OBD_FLAG))       # OBD flag
    )

    # ── Step 12: End of journey flag ──────────────────────────────────────────
    # Some SKUs are flagged for end-of-journey (discontinuation recommendation).
    # Join the latest recommendation for each SKU.
    # Left join preserves all SOQ rows; unmatched SKUs get null flag.
    logger.info("Joining end-of-journey recommendations")

    end_journey = (
        session.table(END_JOURNEY_TABLE)
        .filter(
            # Use only the most recent run of end-of-journey recommendations.
            F.col('RUN_DATE') == session.table(END_JOURNEY_TABLE)
                                        .select(F.max('RUN_DATE'))
                                        .collect()[0][0]
        )
        .select('SKU', 'RECOMMEND_END_OF_JOURNEY')
    )

    data = data.join(end_journey, on='SKU', how='left')

    # ── Step 13: Write output ─────────────────────────────────────────────────
    # Append to SOQ output table — one row per SKU per service level per run.
    # Append (not overwrite) because main() loops over multiple service levels
    # and all results should coexist in the same table.
    row_count = data.count()
    logger.info(f"Writing {row_count} rows to {SOQ_OUTPUT_TABLE}")

    data.write.mode('append').save_as_table(SOQ_OUTPUT_TABLE)

    logger.info(
        f"SOQ calculation complete | month={planning_month} | period={date_period} "
        f"| service_level={service_level}% | rows_written={row_count}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────
def main(session: snowpark.Session):
    """
    Orchestrates SOQ computation across all combinations of:
    - Planning months
    - Stock date periods (first / mid)
    - Service levels (80, 85, 90, 95, 99)
    - Demand variability modes (SKU-level and family-level)

    Each combination is an independent SOQ scenario appended to SOQ_OUTPUT_TABLE.
    Total runs per month × period = 10 (5 service levels × 2 variability modes).
    """

    logger.info(
        f"SOQ pipeline started | months={MONTHS} | periods={STOCK_DATE_TYPE} "
        f"| run_version={RUN_VERSION} | run_date={RUN_DATE}"
    )

    total_runs = len(MONTHS) * len(STOCK_DATE_TYPE) * len(Z_SCORE) * 2
    completed  = 0

    for month in MONTHS:
        for period in STOCK_DATE_TYPE:
            for service_level in Z_SCORE.keys():

                # SKU-level demand variability run
                calculate_soq(
                    session, month, period,
                    service_level=service_level,
                    run_date=RUN_DATE,
                    sku_demand_variability=True,
                    run_version=RUN_VERSION
                )
                completed += 1
                logger.info(f"Progress: {completed}/{total_runs} runs complete")

                # Family-level demand variability run
                calculate_soq(
                    session, month, period,
                    service_level=service_level,
                    run_date=RUN_DATE,
                    sku_demand_variability=False,
                    run_version=RUN_VERSION
                )
                completed += 1
                logger.info(f"Progress: {completed}/{total_runs} runs complete")

    logger.info(f"SOQ pipeline finished | total runs completed: {completed}")

    return session.table(SOQ_OUTPUT_TABLE)
