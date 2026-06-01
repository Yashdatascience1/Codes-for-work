import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import math
import numpy as np
import pandas as pd
import datetime
import logging
import io

# =============================================================================
# CONFIGURATION
# =============================================================================

MONTHS          = ['2026-06-01']
STOCK_DATE_TYPE = ['first']         # options: "first", "end", "mid"
RUN_VERSION     = 33
RUN_DATE        = datetime.datetime.today().strftime('%Y%m%d')
IS_OBD          = True
obd_flag        = 'Y' if IS_OBD else 'N'

# ABC safety-stock days: A-class families get 30 days buffer, B get 25, C get 20
ABC = {"A": 30, "B": 25, "C": 20}

# Pre-format ABC config as a single audit string stored on every output row
STR_ABC = ','.join([f"{k} {v}" for k, v in ABC.items()])

# Z-scores for each service level (one-tailed normal distribution)
Z_SCORE = {95: 1.65, 90: 1.28, 85: 1.04, 80: 0.85, 99: 2.33}

# Service levels and demand variability modes to run in the outer loop
SERVICE_LEVELS         = [95, 90, 85, 99, 80]
DEMAND_VARIABILITY_MODES = [True, False]   # True=SKU-level, False=Family-level

# Table paths
BASE_SOQ_TABLE               = 'MOP_DATABASE.SOQ.SOQ_BASE_TABLE_FINAL_CONCATENATED'
DEMAND_VARIABILITY_TABLE     = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_FINAL_VERSION'
DEMAND_VARIABILITY_FAMILY_TABLE = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_MODEL_FAMILY_FINAL_VERSION'
SOQ_TABLE                    = 'MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_2'
NULL_LEAD_TIME_TABLE         = 'MOP_DATABASE.SOQ.NULL_LEAD_TIME'

END_JOURNEY_QUERY = """
    SELECT SKU, RECOMMEND_END_OF_JOURNEY
    FROM MOP_DATABASE.SOQ.END_OF_JOURNEY_RECOMMENDATION
    WHERE RUN_DATE = (SELECT MAX(RUN_DATE) FROM MOP_DATABASE.SOQ.END_OF_JOURNEY_RECOMMENDATION)
"""


# =============================================================================
# LOGGER SETUP
# =============================================================================

def _setup_logger():
    log_stream = io.StringIO()
    logger = logging.getLogger("SOQ_Step4")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger, log_stream


# =============================================================================
# HELPER: ABC CLASSIFICATION
# =============================================================================

def assign_abc_label(cumulative_percent):
    """
    Maps a cumulative percent-of-sales value to an ABC tier.
    Boundaries match the ABC dict (A=top 70%, B=70-90%, C=tail >90%).
    Uses percentage scale (0-100), not fractional (0-1).
    """
    if cumulative_percent < 70:
        return 'A'
    elif cumulative_percent > 90:
        return 'C'
    else:
        return 'B'


# =============================================================================
# BLOCK A — LOAD SOQ BASE DATA + DEMAND VARIABILITY
# =============================================================================

def load_soq_and_variability(session, planning_month, date_period,
                              sku_demand_variability, run_date, run_version, logger):
    """
    Loads the SOQ base table for the given planning month / stock period / run,
    then merges the appropriate demand variability table (SKU-level or family-level).

    Returns a single merged DataFrame with a DEMAND_VARIABILITY column and a
    DEMAND_VARIABILITY_TYPE label for audit purposes.
    """
    # Pull the SOQ base data for this specific run slice
    soq_query = f"""
        SELECT * FROM {BASE_SOQ_TABLE}
        WHERE PLANNING_MONTH = '{planning_month}'
          AND STOCK_DATE_PERIOD = '{date_period}'
          AND RUN_DATE = {run_date}
          AND RUN_VERSION = {run_version}
          AND IS_OBD = '{obd_flag}'
    """
    soq_data = session.sql(soq_query).to_pandas()
    soq_data.drop(columns=['IS_OBD'], inplace=True)
    logger.info("SOQ base loaded | Month: %s | Period: %s | Rows: %s",
                planning_month, date_period, len(soq_data))

    if sku_demand_variability:
        # SKU-level variability: joined on PARENT_DEALER_CODE + UNIQUE FAMILY CODE + SKU
        dv_query = f"""
            SELECT * FROM {DEMAND_VARIABILITY_TABLE}
            WHERE PLANNING_MONTH = '{planning_month}'
              AND RUN_DATE = {run_date}
              AND RUN_VERSION = {run_version}
              AND IS_OBD = '{obd_flag}'
        """
        demand_variability = session.sql(dv_query).to_pandas()
        demand_variability.drop(columns=['IS_OBD'], inplace=True)

        data = pd.merge(soq_data, demand_variability,
                        on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'SKU', 'PLANNING_MONTH'],
                        how='left')
        data['DEMAND_VARIABILITY_TYPE'] = 'SKU_BASED'
        logger.info("Demand variability type: SKU_BASED | Rows after merge: %s", len(data))
    else:
        # Family-level variability: joined on PARENT_DEALER_CODE + UNIQUE FAMILY CODE only
        dv_query = f"""
            SELECT * FROM {DEMAND_VARIABILITY_FAMILY_TABLE}
            WHERE PLANNING_MONTH = '{planning_month}'
              AND RUN_DATE = {run_date}
              AND RUN_VERSION = {run_version}
              AND IS_OBD = '{obd_flag}'
        """
        demand_variability = session.sql(dv_query).to_pandas()
        demand_variability.drop(columns=['IS_OBD'], inplace=True)

        data = pd.merge(soq_data, demand_variability,
                        on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'PLANNING_MONTH'],
                        how='left')
        data['DEMAND_VARIABILITY_TYPE'] = 'MODEL_SKU_FAMILY_BASED'
        logger.info("Demand variability type: MODEL_SKU_FAMILY_BASED | Rows after merge: %s",
                    len(data))

    # Default variability to 1 for any SKU with no computed history
    null_dv_count = data['DEMAND_VARIABILITY'].isnull().sum()
    data.loc[data['DEMAND_VARIABILITY'].isnull(), 'DEMAND_VARIABILITY'] = 1
    logger.info("Null DEMAND_VARIABILITY filled with 1 | Count: %s", null_dv_count)

    return data


# =============================================================================
# BLOCK B — ABC CLASSIFICATION
# =============================================================================

def compute_abc_classification(data, logger):
    """
    Classifies each (dealer, family) combination into A / B / C based on its
    share of the dealer's total predicted sales.

    Method:
      1. Deduplicate to one row per dealer + family (since data is at SKU level)
      2. Compute each family's % of the dealer's total
      3. Sort descending within each dealer, cumsum the percentages
      4. Assign A (cumulative < 70%), B (70–90%), C (> 90%)

    Returns data with ABC label and SAFETY_STOCK_DAYS column attached.
    """
    # Deduplicate: one row per dealer+family for the sales ranking step
    sales_data = (
        data[['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'PREDICTED_SALES']]
        .drop_duplicates()
        .copy()
    )

    # Dealer-level total (denominator for percentage)
    dealer_totals = (
        sales_data.groupby('PARENT_DEALER_CODE')['PREDICTED_SALES']
        .sum()
        .reset_index()
        .rename(columns={'PREDICTED_SALES': 'DEALER_PREDICTED_SALES'})
    )

    sales_data = pd.merge(sales_data, dealer_totals, on='PARENT_DEALER_CODE', how='left')
    sales_data['PERC_SALES'] = (
        (sales_data['PREDICTED_SALES'] / sales_data['DEALER_PREDICTED_SALES']) * 100
    ).fillna(0)

    # Sort high-to-low within each dealer, then cumsum to get running contribution
    sales_data = sales_data.sort_values(
        by=['PARENT_DEALER_CODE', 'PERC_SALES'], ascending=[True, False]
    )
    sales_data['CUMULATIVE_PERCENT_SALES'] = (
        sales_data.groupby('PARENT_DEALER_CODE')['PERC_SALES'].cumsum()
    )

    sales_data['ABC'] = sales_data['CUMULATIVE_PERCENT_SALES'].apply(assign_abc_label)
    sales_data = sales_data[['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'ABC']]

    data = pd.merge(data, sales_data, on=['PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE'], how='left')

    # Map ABC tier to the corresponding safety stock days from config
    data['SAFETY_STOCK_DAYS'] = data['ABC'].apply(lambda x: ABC.get(x, 15))

    abc_dist = data['ABC'].value_counts().to_dict()
    logger.info("ABC classification complete | Distribution: %s", abc_dist)

    return data


# =============================================================================
# BLOCK C — LEAD TIME STOCK CALCULATION
# =============================================================================

def compute_lead_time_stock(data, session, logger):
    """
    Splits rows with null MAX_LEAD_TIME into a separate audit table, then
    computes how many units of stock are consumed during lead time for each
    of max / min / avg lead time variants.

    Formula: LEAD_TIME_STOCK = ceil( PREDICTED_SALES_SKU * LEAD_TIME_DAYS / 30 )
    (dividing by 30 converts the monthly sales forecast to a daily rate)
    """
    # Separate rows with no transit data — these can't get an SOQ
    null_lt_mask = data['MAX_LEAD_TIME'].isnull()
    null_lt_count = null_lt_mask.sum()

    if null_lt_count > 0:
        null_lead_time_df = data[null_lt_mask].copy()
        session.create_dataframe(null_lead_time_df).write.mode("overwrite").save_as_table(
            NULL_LEAD_TIME_TABLE
        )
        logger.warning("%s rows have null MAX_LEAD_TIME — saved to %s and excluded.",
                       null_lt_count, NULL_LEAD_TIME_TABLE)

    data = data[~null_lt_mask].copy()
    logger.info("Rows retained after lead time null filter: %s", len(data))

    # Lead time stock = units consumed during delivery window
    # ceil() ensures we never under-order due to fractional rounding
    for variant in ['MAX', 'MIN', 'AVG']:
        data[f'{variant}_LEAD_TIME_STOCK'] = (
            (data['PREDICTED_SALES_SKU'] * data[f'{variant}_LEAD_TIME']) / 30
        ).apply(math.ceil)

    return data


# =============================================================================
# BLOCK D — SAFETY STOCK CALCULATION
# =============================================================================

def compute_safety_stock(data, service_level, logger):
    """
    Safety stock = ceil( DEMAND_VARIABILITY * sqrt(SAFETY_STOCK_DAYS / 30) * Z )

    Where:
      DEMAND_VARIABILITY  = std dev of month-over-month sales deltas (from Step 3)
      SAFETY_STOCK_DAYS   = days of buffer determined by ABC tier
      Z                   = z-score for the chosen service level
      Dividing by 30      = converts days to a monthly fraction

    Hard cap: safety stock cannot exceed 3x the monthly predicted sales.
    This prevents absurdly large buffers for highly volatile but low-volume SKUs.
    """
    z = Z_SCORE[service_level]

    raw_ss = data['DEMAND_VARIABILITY'] * np.sqrt(data['SAFETY_STOCK_DAYS'] / 30)
    data['SAFETY_STOCK'] = raw_ss.apply(lambda x: math.ceil(x * z))

    # Cap at 3 months of predicted sales
    cap = data['PREDICTED_SALES_SKU'].fillna(0) * 3
    data['SAFETY_STOCK'] = np.minimum(data['SAFETY_STOCK'].fillna(0), cap)

    logger.info("Safety stock computed | Service level: %s%% | Z-score: %s | "
                "Avg safety stock: %.2f", service_level, z, data['SAFETY_STOCK'].mean())

    return data


# =============================================================================
# BLOCK E — SOQ COMPUTATION (TWO APPROACHES)
# =============================================================================

def compute_soq(data, logger):
    """
    Computes SOQ using two approaches for each of max / min / avg lead times.

    APPROACH 1 (full coverage):
      REORDER_STOCK     = SAFETY_STOCK + LEAD_TIME_STOCK
      TOTAL_STOCK_NEED  = PREDICTED_SALES_SKU + REORDER_STOCK
      SUGGESTED_ORDER   = TOTAL_STOCK_NEED - CURRENT_STOCK
      ADJUSTED_ORDER    = max(0, SUGGESTED_ORDER)   <- can't order negative units

    APPROACH 2 (reorder-point only):
      SOQ               = REORDER_STOCK - CURRENT_STOCK
      ADJUSTED_ORDER    = max(0, SOQ)

    Approach 1 is more conservative (covers both sales AND the reorder buffer).
    Approach 2 only covers the buffer, assuming sales will be covered separately.
    """
    # Fill null stock with 0 (dealer has no physical inventory on hand)
    null_stk = data['STK_AS_ON_DATE'].isnull().sum()
    data.loc[data['STK_AS_ON_DATE'].isnull(), 'STK_AS_ON_DATE'] = 0
    if null_stk:
        logger.info("Null STK_AS_ON_DATE filled with 0 | Count: %s", null_stk)

    for variant in ['MAX', 'MIN', 'AVG']:
        lt_stock = data[f'{variant}_LEAD_TIME_STOCK']
        ss       = data['SAFETY_STOCK']
        pred     = data['PREDICTED_SALES_SKU']
        stk      = data['STK_AS_ON_DATE']

        # Approach 1
        reorder            = ss + lt_stock
        total_need         = pred + reorder
        suggested          = total_need - stk
        data[f'{variant}_REORDER_STOCK']            = reorder
        data[f'{variant}_TOTAL_STOCK_SKU']          = total_need
        data[f'{variant}_Suggested_Stock_SKU']      = suggested
        data[f'{variant}_Adjusted_Monthly_Order']   = suggested.apply(lambda x: max(0, x))

        # Approach 2
        soq2                                        = reorder - stk
        data[f'{variant}_SOQ_APPROACH_2']           = soq2
        data[f'{variant}_Adjusted_Monthly_Order_APPROACH_2'] = soq2.apply(lambda x: max(0, x))

    logger.info("SOQ computed (both approaches) for MAX / MIN / AVG lead times.")
    return data


# =============================================================================
# BLOCK F — END-OF-JOURNEY FLAG + METADATA + SAVE
# =============================================================================

def attach_metadata_and_save(session, data, service_level, run_date, run_version, logger):
    """
    Attaches audit/metadata columns and the end-of-journey flag, then appends
    the result to the final SOQ table.
    """
    # Overwrite ABC column with the human-readable config string for audit
    data['ABC']           = STR_ABC
    data['Z_SCORE']       = Z_SCORE[service_level]
    data['SERVICE_LEVEL'] = service_level
    data['RUN_DATE']      = run_date
    data['RUN_VERSION']   = run_version
    data['IS_OBD']        = obd_flag

    # End-of-journey flag: marks SKUs the business recommends phasing out
    end_journey = session.sql(END_JOURNEY_QUERY).to_pandas()
    data = pd.merge(data, end_journey, on='SKU', how='left')
    logger.info("End-of-journey flag joined | SKUs flagged: %s",
                data['RECOMMEND_END_OF_JOURNEY'].notna().sum())

    session.create_dataframe(data).write.mode("append").save_as_table(SOQ_TABLE)
    logger.info("SOQ data appended to %s | Rows written: %s", SOQ_TABLE, len(data))


# =============================================================================
# ORCHESTRATOR: calculateSOQ
# =============================================================================

def calculateSOQ(session, planning_month, date_period, service_level,
                 run_date, sku_demand_variability, run_version, logger):
    """
    Full SOQ calculation pipeline for one combination of:
      planning_month, date_period, service_level, demand_variability_mode.
    """
    logger.info("--- calculateSOQ | Month: %s | Period: %s | SL: %s%% | DV mode: %s ---",
                planning_month, date_period, service_level,
                "SKU" if sku_demand_variability else "FAMILY")

    data = load_soq_and_variability(
        session, planning_month, date_period,
        sku_demand_variability, run_date, run_version, logger
    )
    data = compute_abc_classification(data, logger)
    data = compute_lead_time_stock(data, session, logger)
    data = compute_safety_stock(data, service_level, logger)
    data = compute_soq(data, logger)
    attach_metadata_and_save(session, data, service_level, run_date, run_version, logger)


# =============================================================================
# MAIN
# =============================================================================

def main(session: snowpark.Session):
    logger, log_stream = _setup_logger()
    logger.info("========== SOQ Step 4 Pipeline Started ==========")

    try:
        for planning_month in MONTHS:
            for date_period in STOCK_DATE_TYPE:
                for service_level in SERVICE_LEVELS:
                    for sku_dv in DEMAND_VARIABILITY_MODES:
                        calculateSOQ(
                            session, planning_month, date_period,
                            service_level, RUN_DATE, sku_dv, RUN_VERSION, logger
                        )

        logger.info("========== SOQ Step 4 Pipeline Complete ==========")

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        raise

    finally:
        print(log_stream.getvalue())

    return session.table(SOQ_TABLE)
