import snowflake.snowpark as snowpark
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
import logging
import datetime
import os

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Logging
# Always configure at module level before anything else.
# MODULE_NAME uses the filename so logs are identifiable when running
# multiple scripts in sequence and storing logs to Snowflake stage.
# ─────────────────────────────────────────────────────────────────────────────
MODULE_NAME = os.path.splitext(os.path.basename(__file__))[0]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)-25s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(MODULE_NAME)

# Optional: write logs to a local file and push to Snowflake stage at end of run.
# Uncomment the block below to enable file logging.
# RUN_DATE_FOR_LOG = datetime.datetime.today().strftime('%Y%m%d')
# _file_handler = logging.FileHandler(f'/tmp/{MODULE_NAME}_{RUN_DATE_FOR_LOG}.log')
# _file_handler.setFormatter(logging.Formatter(
#     '%(asctime)s  %(name)-25s  %(levelname)-8s  %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# ))
# logger.addHandler(_file_handler)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CONFIG
# All tunable parameters in one place.
# Change values here only — never hardcode inside functions.
# ─────────────────────────────────────────────────────────────────────────────
MONTHS          = ['2026-06-01']        # planning months to compute SOQ for
STOCK_DATE_TYPE = ['first']             # 'first' or 'mid' — stock snapshot timing
RUN_VERSION     = 33                    # increment each time pipeline is re-run
RUN_DATE        = datetime.datetime.today().strftime('%Y%m%d')  # auto-derived today

# ABC classification → safety stock days
# A = high-value families (30 days stock), B = mid (25), C = tail (20)
ABC = {'A': 30, 'B': 25, 'C': 20}

# Z-scores per service level — controls how many std deviations safety stock covers
Z_SCORE = {95: 1.65, 90: 1.28, 85: 1.04, 80: 0.85, 99: 2.33}

# Human-readable ABC config string — stored in output for auditability
STR_ABC = ', '.join([f'{k} {v}' for k, v in ABC.items()])

# OBD flag — whether this run uses OBD-mapped SKUs
IS_OBD   = True
OBD_FLAG = 'Y' if IS_OBD else 'N'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Table references
# All table names centralised here. Change once if tables are renamed.
# ─────────────────────────────────────────────────────────────────────────────
BASE_SOQ_TABLE               = 'MOP_DATABASE.SOQ.SOQ_BASE_TABLE_FINAL_CONCATENATED'
DEMAND_VARIABILITY_TABLE     = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_FINAL_VERSION'
DEMAND_VARIABILITY_FAM_TABLE = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_MODEL_FAMILY_FINAL_VERSION'
SOQ_OUTPUT_TABLE             = 'MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_2'
NULL_LEAD_TIME_TABLE         = 'MOP_DATABASE.SOQ.NULL_LEAD_TIME'
END_JOURNEY_TABLE            = 'MOP_DATABASE.SOQ.END_OF_JOURNEY_RECOMMENDATION'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — ABC helper function
# Pure Python function — no Snowpark dependencies.
# Registered as a UDF in Step 8 so it runs server-side in Snowflake.
# ─────────────────────────────────────────────────────────────────────────────
def _get_abc_class(cumulative_pct: float) -> str:
    """
    Classifies a dealer-family into ABC based on cumulative % of dealer sales.
    A: top revenue families  (cumulative_pct < 70)
    B: mid-tier families     (70 <= cumulative_pct <= 90)
    C: tail families         (cumulative_pct > 90)
    """
    if cumulative_pct < 70:
        return 'A'
    elif cumulative_pct > 90:
        return 'C'
    else:
        return 'B'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — calculate_soq function signature
# Typed parameters with defaults. One call = one combination of
# planning month × date period × service level × variability mode.
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
    Computes Suggested Order Quantity (SOQ) for one scenario and appends
    the result to SOQ_OUTPUT_TABLE.

    Parameters
    ----------
    planning_month         : First day of the planning month (YYYY-MM-DD).
    date_period            : 'first' or 'mid' — stock snapshot timing.
    service_level          : Service level % — drives Z-score for safety stock.
    run_date               : YYYYMMDD string used as partition key in output.
    sku_demand_variability : True → SKU-level variability; False → family-level.
    run_version            : Integer version tag for this run.
    """
    logger.info(
        f"Starting SOQ | month={planning_month} | period={date_period} "
        f"| sl={service_level}% | sku_var={sku_demand_variability} | v={run_version}"
    )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — Load and filter base SOQ table
    # BASE_SOQ_TABLE has forecasts + stock positions per dealer-family-SKU.
    # Apply all partition filters immediately so downstream steps are lean.
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Loading base SOQ table")

    soq_base = (
        session.table(BASE_SOQ_TABLE)
        .filter(F.col('PLANNING_MONTH')    == F.lit(planning_month))
        .filter(F.col('STOCK_DATE_PERIOD') == F.lit(date_period))
        .filter(F.col('RUN_DATE')          == F.lit(int(run_date)))  # stored as int in table
        .filter(F.col('RUN_VERSION')       == F.lit(run_version))
        .filter(F.col('IS_OBD')            == F.lit(OBD_FLAG))
        .drop('IS_OBD')   # filtered on; not needed further downstream
    )

    logger.info(f"Base SOQ rows loaded: {soq_base.count()}")


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — Load demand variability and join onto base data
    # Demand variability = how much a SKU's/family's sales fluctuate.
    # Higher variability → larger safety stock needed.
    # Branch on sku_demand_variability:
    #   True  → join at SKU level (more granular)
    #   False → join at family level (all SKUs in family share same variability)
    # Left join: unmatched rows get null variability → coalesced to 1.0 (neutral).
    # ─────────────────────────────────────────────────────────────────────────
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

        # SKU is included in join key — each SKU gets its own variability value
        data = soq_base.join(
            demand_var,
            on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'SKU', 'PLANNING_MONTH'],
            how='left'
        )

        # Null variability → 1.0 (neutral multiplier, no safety stock amplification)
        data = data.with_column(
            'DEMAND_VARIABILITY',
            F.coalesce(F.col('DEMAND_VARIABILITY'), F.lit(1.0))
        )

        # Tag for auditability — visible in output table
        data = data.with_column('DEMAND_VARIABILITY_TYPE', F.lit('SKU_BASED'))

    else:
        logger.info("Loading family-level demand variability")

        demand_var = (
            session.table(DEMAND_VARIABILITY_FAM_TABLE)
            .filter(F.col('PLANNING_MONTH') == F.lit(planning_month))
            .filter(F.col('RUN_DATE')       == F.lit(int(run_date)))
            .filter(F.col('RUN_VERSION')    == F.lit(run_version))
            .filter(F.col('IS_OBD')         == F.lit(OBD_FLAG))
            .drop('IS_OBD')
        )

        # SKU NOT in join key — all SKUs within a family inherit the family variability
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

    logger.info(f"Demand variability joined | rows: {data.count()}")


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8 — ABC classification
    # Classifies each dealer-family by its revenue contribution within its dealer.
    # Used in Step 9 to assign safety stock days.
    #
    # Sub-step 8a: deduplicate to dealer-family level and compute % share.
    # Deduplication avoids double-counting when multiple SKU rows exist per family.
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Computing ABC classification")

    # Window over dealer — used to compute dealer-level total sales
    dealer_window = Window.partition_by('PARENT_DEALER_CODE')

    abc_data = (
        data
        .select('PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'PREDICTED_SALES')
        .distinct()   # one row per dealer-family; prevents SKU-level double counting

        # Dealer total = sum of all family sales within this dealer
        .with_column(
            'DEALER_PREDICTED_SALES',
            F.sum('PREDICTED_SALES').over(dealer_window)
        )

        # % share of this family within its dealer's total predicted sales
        # Guard against division by zero — assign 0% if dealer has no sales
        .with_column(
            'PERC_SALES',
            F.when(
                F.col('DEALER_PREDICTED_SALES') == F.lit(0), F.lit(0.0)
            ).otherwise(
                (F.col('PREDICTED_SALES') / F.col('DEALER_PREDICTED_SALES')) * F.lit(100)
            )
        )
    )

    # Sub-step 8b: compute cumulative % sorted by sales descending within each dealer.
    # Sorted descending so highest-revenue families accumulate first → land in A.
    family_window = (
        Window.partition_by('PARENT_DEALER_CODE')
              .order_by(F.col('PREDICTED_SALES').desc())
    )

    abc_data = abc_data.with_column(
        'CUMULATIVE_PERCENT_SALES',
        F.sum('PERC_SALES').over(
            family_window.rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
    )

    # Sub-step 8c: register ABC helper as UDF and apply.
    # UDF needed because the conditional logic cannot be expressed as a
    # native Snowpark expression. Runs server-side in Snowflake.
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
        .select('PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'ABC')  # only columns needed downstream
    )

    # Join ABC class back onto main dataset at dealer-family level
    data = data.join(abc_data, on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE'], how='left')

    logger.info("ABC classification joined")


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9 — Safety stock days from ABC class
    # A-class families get 30 days — high revenue, stockouts are costly.
    # C-class families get 20 days — tail families, lower impact.
    # Default 15 days for any row that couldn't be classified.
    # ─────────────────────────────────────────────────────────────────────────
    data = data.with_column(
        'SAFETY_STOCK_DAYS',
        F.when(F.col('ABC') == F.lit('A'), F.lit(ABC['A']))
         .when(F.col('ABC') == F.lit('B'), F.lit(ABC['B']))
         .when(F.col('ABC') == F.lit('C'), F.lit(ABC['C']))
         .otherwise(F.lit(15))   # fallback — should not trigger if ABC join is complete
    )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 10 — Handle null lead time rows
    # Lead time is required for every SOQ computation.
    # Rows missing MAX_LEAD_TIME are saved for investigation then dropped.
    # Overwrite null lead time table each run — it reflects current state only.
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Checking for null lead time rows")

    null_rows  = data.filter(F.col('MAX_LEAD_TIME').is_null())
    null_count = null_rows.count()

    if null_count > 0:
        logger.warning(
            f"{null_count} rows with null MAX_LEAD_TIME — "
            f"saving to {NULL_LEAD_TIME_TABLE} and dropping from SOQ"
        )
        null_rows.write.mode('overwrite').save_as_table(NULL_LEAD_TIME_TABLE)
    else:
        logger.info("No null lead time rows found")

    # Drop null lead time rows — SOQ is meaningless without lead time
    data = data.filter(F.col('MAX_LEAD_TIME').is_not_null())

    logger.info(f"Rows after null lead time drop: {data.count()}")


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 11 — Lead time stock
    # How much stock is consumed while waiting for replenishment.
    # Formula: (PREDICTED_SALES_SKU × LEAD_TIME_DAYS) / 30
    # Divide by 30: converts monthly sales to a daily rate, then scales to lead time.
    # F.ceil: always round up — never under-stock due to rounding.
    # Three variants: MAX (worst case), MIN (best case), AVG (balanced).
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
    # STEP 12 — Safety stock
    # Protects against demand variability during the safety stock coverage period.
    #
    # Formula: DEMAND_VARIABILITY × sqrt(SAFETY_STOCK_DAYS / 30) × Z_SCORE
    #
    # DEMAND_VARIABILITY     : std dev of demand — higher = more volatile SKU
    # sqrt(SS_DAYS / 30)     : scales variability to the coverage horizon.
    #                          Square root because demand variance compounds
    #                          sub-linearly over independent time periods.
    # Z_SCORE                : from service level — 95% → 1.65 std devs of coverage
    #
    # Cap at 3× predicted monthly sales: prevents extreme safety stock on
    # very volatile but low-volume SKUs where formula can explode.
    # F.ceil: always round up.
    # Coalesce to 0: safe fallback for any remaining nulls after the cap.
    # ─────────────────────────────────────────────────────────────────────────
    z = Z_SCORE[service_level]   # resolve Z-score for this service level

    data = data.with_column(
        'SAFETY_STOCK',
        F.least(
            # Uncapped formula
            F.ceil(
                F.col('DEMAND_VARIABILITY')
                * F.sqrt(F.col('SAFETY_STOCK_DAYS') / F.lit(30))
                * F.lit(z)
            ),
            # Cap: 3× monthly predicted SKU sales
            F.col('PREDICTED_SALES_SKU') * F.lit(3)
        )
    )

    # Null fallback — coalesce after the cap in case PREDICTED_SALES_SKU was null
    data = data.with_column(
        'SAFETY_STOCK',
        F.coalesce(F.col('SAFETY_STOCK'), F.lit(0))
    )

    # Null stock on date → treat as zero (no stock at dealer)
    data = data.with_column(
        'STK_AS_ON_DATE',
        F.coalesce(F.col('STK_AS_ON_DATE'), F.lit(0))
    )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 13 — SOQ Approach 1: full month coverage
    # Suggests enough stock to cover predicted demand + safety buffer + lead time.
    # Then subtracts existing stock on hand — only order what's actually needed.
    # F.greatest(..., 0): floors at zero — never suggest a negative order.
    #
    # Three scenarios per approach — MAX/MIN/AVG lead time:
    #   MAX: conservative — assumes supplier takes longest possible time
    #   MIN: aggressive   — assumes supplier delivers fastest
    #   AVG: balanced     — uses historical average lead time
    # ─────────────────────────────────────────────────────────────────────────
    data = (
        data
        # MAX lead time scenario
        .with_column('MAX_REORDER_STOCK',        # buffer = safety stock + lead time stock
                     F.col('SAFETY_STOCK') + F.col('MAX_LEAD_TIME_STOCK'))
        .with_column('MAX_TOTAL_STOCK_SKU',      # total needed = demand + buffer
                     F.col('PREDICTED_SALES_SKU') + F.col('MAX_REORDER_STOCK'))
        .with_column('MAX_Suggested_Stock_SKU',  # suggested order = total needed - on hand
                     F.col('MAX_TOTAL_STOCK_SKU') - F.col('STK_AS_ON_DATE'))
        .with_column('MAX_Adjusted_Monthly_Order',  # floor at 0
                     F.greatest(F.col('MAX_Suggested_Stock_SKU'), F.lit(0)))

        # MIN lead time scenario
        .with_column('MIN_REORDER_STOCK',
                     F.col('SAFETY_STOCK') + F.col('MIN_LEAD_TIME_STOCK'))
        .with_column('MIN_TOTAL_STOCK_SKU',
                     F.col('PREDICTED_SALES_SKU') + F.col('MIN_REORDER_STOCK'))
        .with_column('MIN_Suggested_Stock_SKU',
                     F.col('MIN_TOTAL_STOCK_SKU') - F.col('STK_AS_ON_DATE'))
        .with_column('MIN_Adjusted_Monthly_Order',
                     F.greatest(F.col('MIN_Suggested_Stock_SKU'), F.lit(0)))

        # AVG lead time scenario
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
    # STEP 14 — SOQ Approach 2: reorder point only
    # Leaner alternative: replenish only to the reorder point
    # (safety stock + lead time stock), not the full month's predicted demand.
    # Appropriate when the dealer already holds sufficient forward stock.
    # Formula: REORDER_STOCK - STK_AS_ON_DATE (floored at 0).
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
    # STEP 15 — Audit columns
    # Tag every output row with run metadata.
    # Done last — after all business logic — so tags don't interfere with joins.
    # These columns allow any output row to be traced back to the exact
    # run configuration that produced it.
    # ─────────────────────────────────────────────────────────────────────────
    data = (
        data
        .with_column('ABC',           F.lit(STR_ABC))       # ABC thresholds used
        .with_column('Z_SCORE',       F.lit(z))             # Z-score for this service level
        .with_column('SERVICE_LEVEL', F.lit(service_level)) # service level %
        .with_column('RUN_DATE',      F.lit(run_date))      # date of this run
        .with_column('RUN_VERSION',   F.lit(run_version))   # version tag
        .with_column('IS_OBD',        F.lit(OBD_FLAG))      # OBD flag
    )


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 16 — End of journey join
    # Some SKUs are flagged for discontinuation. Join the latest recommendation.
    # Filter to max RUN_DATE inside the table — always uses the freshest data.
    # Left join: SOQ rows without an end-of-journey record get null flag.
    # Done last because it is enrichment only — not used in any calculation.
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Joining end-of-journey recommendations")

    # Collect max run date from end-of-journey table — single row, cheap operation
    max_eoj_run_date = (
        session.table(END_JOURNEY_TABLE)
               .select(F.max('RUN_DATE'))
               .collect()[0][0]
    )

    end_journey = (
        session.table(END_JOURNEY_TABLE)
        .filter(F.col('RUN_DATE') == F.lit(max_eoj_run_date))
        .select('SKU', 'RECOMMEND_END_OF_JOURNEY')
    )

    data = data.join(end_journey, on='SKU', how='left')


    # ─────────────────────────────────────────────────────────────────────────
    # STEP 17 — Write output
    # Append — not overwrite — because main() calls this function 10 times
    # per month × period combination (5 service levels × 2 variability modes).
    # All 10 runs must coexist in the same output table.
    # Log row count before writing for observability.
    # ─────────────────────────────────────────────────────────────────────────
    row_count = data.count()
    logger.info(f"Writing {row_count} rows to {SOQ_OUTPUT_TABLE}")

    data.write.mode('append').save_as_table(SOQ_OUTPUT_TABLE)

    logger.info(
        f"Done | month={planning_month} | period={date_period} "
        f"| sl={service_level}% | rows_written={row_count}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 18 — main: orchestrator
# Loops over all combinations and calls calculate_soq for each.
# 10 runs per month × period: 5 service levels × 2 variability modes.
# Progress counter tells you exactly where the pipeline is at any point.
# ─────────────────────────────────────────────────────────────────────────────
def main(session: snowpark.Session):
    """
    Entry point. Orchestrates SOQ computation across all combinations of:
      - Planning months
      - Stock date periods
      - Service levels (80, 85, 90, 95, 99)
      - Demand variability modes (SKU-level and family-level)
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

                # Run 1 of 2 for this service level: SKU-level variability
                calculate_soq(
                    session, month, period,
                    service_level=service_level,
                    run_date=RUN_DATE,
                    sku_demand_variability=True,
                    run_version=RUN_VERSION
                )
                completed += 1
                logger.info(f"Progress: {completed}/{total_runs} runs complete")

                # Run 2 of 2 for this service level: family-level variability
                calculate_soq(
                    session, month, period,
                    service_level=service_level,
                    run_date=RUN_DATE,
                    sku_demand_variability=False,
                    run_version=RUN_VERSION
                )
                completed += 1
                logger.info(f"Progress: {completed}/{total_runs} runs complete")

    logger.info(f"SOQ pipeline finished | total runs: {completed}/{total_runs}")

    # Optional: push log file to Snowflake stage if file logging is enabled
    # session.file.put(
    #     f'/tmp/{MODULE_NAME}_{RUN_DATE}.log',
    #     f'@MOP_DATABASE.SOQ.LOGS_STAGE/{MODULE_NAME}/',
    #     overwrite=True
    # )

    return session.table(SOQ_OUTPUT_TABLE)