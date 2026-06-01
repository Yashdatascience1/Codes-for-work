import snowflake.snowpark as snowpark
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
import datetime
import logging
import io

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Forecast & prediction tables ---
FORECAST_TABLE          = 'MOP_DATABASE.SOQ.TEST_FORECASTS_WITH_MARKET_SHARE_SON_2025_VIEW_V2'
TFT_FORECAST_TABLE      = 'MOP_DATABASE.SOQ.PREDICTIONS_BY_TFT_JAN_26_TO_APR_26'
TEST_FORECAST_TABLE     = 'MOP_DATABASE.SOQ.TEST_FORECASTS_PARENT_DEALER_MODEL_FAMILY_WITH_MARKET_SHARE_SON_2025'
TEST_DATA_TABLE         = 'MOP_DATABASE.SOQ.TEST_DATA_MONTHLY_DEALER_MODEL_FAMILY_CODE_SON_2025_UPDATED'
PREDICTION_TABLE        = 'MOP_DATABASE.SOQ.SOQ_PREDICTION_FINAL_VERSION'

# --- Output tables ---
OUTPUT_TABLE                  = 'MOP_DATABASE.SOQ.DEALER_SKU_DISAGGREGATION_RESULTS'
BASE_SOQ_TABLE                = 'MOP_DATABASE.SOQ.SOQ_BASE_TABLE_FINAL_CONCATENATED'
DEMAND_VARIABILITY_TABLE      = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_FINAL_VERSION'
DEMAND_VARIABILITY_FAMILY_TABLE = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_MODEL_FAMILY_FINAL_VERSION'

# --- Source / lookup tables ---
ECR_TABLE               = 'ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS'
SKU_SUPERCEDENCE_TABLE  = 'MOP_DATABASE.SOQ.SKU_SUPERCEDENCE_MODEL_FAMILY_MAY_2026_UPDATED_V2'
OBD_MAPPING_TABLE       = 'MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW'
PARENT_DEALER_VIEW      = 'FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH'
TRANSIT_TABLE           = 'MOP_DATABASE.SOQ.PARENT_DEALER_TRANSIT_TIME_SKU_NEW'
STOCK_TABLE             = 'ANALYTICS_DATABASE.ANALYTICS_SALES.STOCK_AVAILABILITY'

# --- Run parameters ---
CUSTOMER_TYPE           = ('Individual',)
IS_OBD                  = True
LOOKBACK_MONTHS         = 3        # months of ECR history used for SKU proportion calculation
MID_DATE                = 15       # day-of-month used when stock_period = "mid"
RUN_DATE                = datetime.datetime.today().strftime('%Y%m%d')
RUN_VERSION             = 33
PRED_LEVEL              = "monthly"

# Planning months and stock date types to loop over
MONTHS          = ['2026-06-01']
STOCK_DATE_TYPE = ['first']        # options: "first", "end", "mid"

obd_flag = 'Y' if IS_OBD else 'N'


# =============================================================================
# HELPER: logging setup
# =============================================================================

def _setup_logger():
    log_stream = io.StringIO()
    logger = logging.getLogger("SOQ_Step3")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger, log_stream


# =============================================================================
# BLOCK A — REFERENCE DATA LOADERS
# =============================================================================

def load_parent_dealer_mapping(session, logger):
    """
    Returns a Snowpark DataFrame: DEALER_CODE -> PARENT_DEALER_CODE.
    PARENT_DEALER_CODE is extracted from PAR_ORG_NAME by splitting on '-' and
    taking the first token (e.g. "DELHI-NORTH" -> "DELHI").
    """
    df = (
        session.table(PARENT_DEALER_VIEW)
        .filter(~F.col("X_DEALER_CODE_HIER").is_null())
        .select(
            F.col("X_DEALER_CODE_HIER").alias("DEALER_CODE"),
            F.col("PAR_ORG_NAME")
        )
        .distinct()
        .with_column("PARENT_DEALER_CODE",
                     F.trim(F.split(F.col("PAR_ORG_NAME"), F.lit('-'))[0]))
        .select("DEALER_CODE", "PARENT_DEALER_CODE")
    )
    logger.info("Parent dealer mapping loaded. Unique dealer codes: %s", df.count())
    return df


def load_sku_supercedence(session, logger):
    """
    Loads the SKU supercedence table, parses UPDATED_ON from YYYYMMDD integer
    to a proper date, logs the latest update date, then drops UPDATED_ON.
    Returns active SKUs only and the count-of-active-SKUs-per-family DataFrame.
    """
    sku_raw = session.table(SKU_SUPERCEDENCE_TABLE)

    # Parse UPDATED_ON from its YYYYMMDD integer storage format
    sku_raw = sku_raw.with_column(
        "UPDATED_ON",
        F.to_date(F.col("UPDATED_ON").cast("string"), F.lit("YYYYMMDD"))
    )
    latest_update = sku_raw.select(F.max("UPDATED_ON")).collect()[0][0]
    logger.info("SKU supercedence table: %s | Latest UPDATED_ON: %s",
                SKU_SUPERCEDENCE_TABLE, latest_update)

    sku_raw = sku_raw.drop("UPDATED_ON")

    # Strip double-quotes from column names (Snowpark join artefact)
    for old_col in sku_raw.columns:
        new_col = old_col.replace('"', '')
        sku_raw = sku_raw.rename(old_col, new_col)

    # Rename the spaced column to an underscore version for clean joins
    sku_raw = sku_raw.with_column_renamed("UNIQUE FAMILY CODE", "UNIQUE_FAMILY_CODE")

    # Keep only active SKUs for disaggregation and proportion calculations
    active_skus = sku_raw.filter(F.col("SKUSTATUS") == F.lit("active"))

    # Count of active SKUs per family — used to do equal-split when history is zero
    num_active = (
        active_skus
        .group_by("UNIQUE_FAMILY_CODE")
        .agg(F.count_distinct("SKU").alias("NUM_ACTIVE_SKUS"))
    )

    unique_families = num_active.select(
        F.count_distinct(F.col("UNIQUE_FAMILY_CODE"))
    ).collect()[0][0]
    total_active = num_active.agg(F.sum("NUM_ACTIVE_SKUS")).collect()[0][0]
    logger.info("Active SKUs: %s across %s unique family codes", total_active, unique_families)

    return active_skus, num_active


# =============================================================================
# BLOCK B — STOCK DATA
# =============================================================================

def get_stock_date(date_period, planning_month):
    """
    Derives the stock snapshot date from the planning month and period type.
      "first" -> last day of the month BEFORE planning_month
                 (i.e. opening stock at start of planning month)
      "end"   -> last day of planning_month itself (closing stock)
      "mid"   -> MID_DATE of the month before planning_month
    """
    from dateutil.relativedelta import relativedelta
    run_month = datetime.datetime.strptime(planning_month, "%Y-%m-%d") - relativedelta(months=1)
    if date_period == "end":
        return (datetime.datetime.strptime(planning_month, "%Y-%m-%d").replace(day=1)
                - relativedelta(days=1)).strftime("%Y-%m-%d")
    if date_period == "first":
        return (run_month.replace(day=1) - relativedelta(days=1)).strftime("%Y-%m-%d")
    if date_period == "mid":
        return run_month.replace(day=MID_DATE).strftime("%Y-%m-%d")


def load_stock_data(session, stock_date, logger):
    """
    Loads dealer-level stock as of stock_date from STOCK_AVAILABILITY.
    If IS_OBD=True, maps old SKU variants to their current OBD SKU and
    sums stock across variants so each current SKU shows total available stock.
    """
    stock_raw = (
        session.table(STOCK_TABLE)
        .filter(F.col("CAL_DATE") == F.lit(stock_date))
        .select("DEALER_CODE", "MODEL", "SKU",
                F.col("CLOSING_STOCK").alias("STK_AS_ON_DATE"))
    )

    if IS_OBD:
        obd = (
            session.table(OBD_MAPPING_TABLE)
            .select(
                F.col("PREVIOUS_OBD_SKU").alias("SKU"),
                F.col("CURRENT_OBD_SKU")
            )
        )
        stock_raw = (
            stock_raw
            .join(obd, on="SKU", how="left")
            .with_column("SKU", F.coalesce(F.col("CURRENT_OBD_SKU"), F.col("SKU")))
            .drop("CURRENT_OBD_SKU")
            # Re-aggregate: sum stock across old+new variant of the same OBD SKU
            .group_by("DEALER_CODE", "MODEL", "SKU")
            .agg(F.sum("STK_AS_ON_DATE").alias("STK_AS_ON_DATE"))
        )

    logger.info("Stock data loaded for date: %s | Rows: %s", stock_date, stock_raw.count())
    return stock_raw


# =============================================================================
# BLOCK C — ECR SALES DATA
# =============================================================================

def load_ecr_data(session, start_date, end_date, logger):
    """
    Pulls individual retail sales (ECR) between start_date and end_date,
    for Individual customers only. Computes NET_SALES and enriches with
    SKU supercedence (family code) and parent dealer mapping.
    If IS_OBD=True, remaps sold SKUs to their current OBD variant.
    """
    ecr_raw = (
        session.table(ECR_TABLE)
        .filter(F.col("X_CUSTOMER_TYPE").isin(list(CUSTOMER_TYPE)))
        .filter(F.col("CAL_DATE") >= F.lit(start_date))
        .filter(F.col("CAL_DATE") <  F.lit(end_date))
        .with_column("CAL_DATE", F.to_date(F.col("CAL_DATE")))
        .with_column(
            "NET_SALES",
            F.col("INVOICED_SALES") + F.col("CANCELLED_SALES") + F.col("RETURNED_SALES")
        )
    )

    # Attach family codes from SKU supercedence
    sku_ref = (
        session.table(SKU_SUPERCEDENCE_TABLE)
        .drop("UPDATED_ON")
    )
    for old_col in sku_ref.columns:
        sku_ref = sku_ref.rename(old_col, old_col.replace('"', ''))
    sku_ref = sku_ref.with_column_renamed("UNIQUE FAMILY CODE", "UNIQUE_FAMILY_CODE")
    sku_ref = sku_ref.select("SKU", "MODEL", "UNIQUE_FAMILY_CODE")

    # Attach parent dealer mapping
    parent_map = (
        session.table(PARENT_DEALER_VIEW)
        .filter(~F.col("X_DEALER_CODE_HIER").is_null())
        .select(
            F.col("X_DEALER_CODE_HIER").alias("DEALER_CODE"),
            F.trim(F.split(F.col("PAR_ORG_NAME"), F.lit('-'))[0]).alias("PARENT_DEALER_CODE")
        )
        .distinct()
    )

    ecr = (
        ecr_raw
        .join(sku_ref,   on=["SKU", "MODEL"], how="left")
        .join(parent_map, on="DEALER_CODE",    how="left")
    )

    # Remap sold SKU to current OBD variant if enabled
    if IS_OBD:
        obd = (
            session.table(OBD_MAPPING_TABLE)
            .select(
                F.col("PREVIOUS_OBD_SKU").alias("SKU"),
                F.col("CURRENT_OBD_SKU")
            )
        )
        ecr = (
            ecr
            .join(obd, on="SKU", how="left")
            .with_column("SKU", F.coalesce(F.col("CURRENT_OBD_SKU"), F.col("SKU")))
            .drop("CURRENT_OBD_SKU")
        )

    logger.info("ECR loaded | Period: %s to %s | Rows: %s", start_date, end_date, ecr.count())
    return ecr


# =============================================================================
# BLOCK D — FORECAST TABLE PROCESSING
# =============================================================================

def process_forecast_table(session, planning_month, logger):
    """
    Reads the TFT forecast table and parses PARENT_DEALER_CODE and
    UNIQUE_FAMILY_CODE out of the composite PARENT_DEALER_CODE_MODEL_FAMILY key.
    The composite key format is: "DEALER<>MODEL_FAMILY<>ATTR1<>ATTR2..."
    """
    forecast = (
        session.table(TFT_FORECAST_TABLE)
        .with_column("MONTH_OF_SALE", F.to_date(F.col("MONTH_OF_SALE")))
        .filter(F.col("MONTH_OF_SALE") == F.lit(planning_month))
    )

    # Extract PARENT_DEALER_CODE: everything before the first '<>'
    forecast = forecast.with_column(
        "PARENT_DEALER_CODE",
        F.trim(F.split(F.col("PARENT_DEALER_CODE_MODEL_FAMILY"), F.lit('<>'))[0])
    )

    # Extract UNIQUE_FAMILY_CODE: everything after the first '<>'
    forecast = forecast.with_column(
        "UNIQUE_FAMILY_CODE",
        F.substr(
            F.col("PARENT_DEALER_CODE_MODEL_FAMILY"),
            F.charindex(F.lit('<>'), F.col("PARENT_DEALER_CODE_MODEL_FAMILY")) + F.lit(2)
        )
    )

    row_count = forecast.count()
    unique_dealers = forecast.select(
        F.count_distinct("PARENT_DEALER_CODE")
    ).collect()[0][0]
    logger.info("Forecast table loaded | Month: %s | Rows: %s | Unique dealers: %s",
                planning_month, row_count, unique_dealers)

    forecast.write.mode("append").save_as_table(PREDICTION_TABLE)
    logger.info("Forecast written to %s", PREDICTION_TABLE)
    return forecast


# =============================================================================
# BLOCK E — SKU DISAGGREGATION
# Splits the family-level TFT forecast down to individual active SKUs
# using each SKU's share of the last LOOKBACK_MONTHS of actual sales.
# =============================================================================

def disaggregate_forecast(session, forecast, active_skus, num_active,
                           planning_month, logger):
    """
    For each (dealer, family, month) row in the forecast:
      1. Pull last LOOKBACK_MONTHS of ECR sales per SKU
      2. Calculate each SKU's proportion of the family total
      3. Multiply proportion by PREDICTED_SALES_TFT to get SKU-level forecast

    Proportion rules (matching Step3.py percentsku logic):
      - No history at all for this family  -> equal split (1 / NUM_ACTIVE_SKUS)
      - Family has history but SKU is zero -> proportion = 0 (SKU gets nothing)
      - Otherwise                          -> DEALER_SKU_SALES / FAMILY_TOTAL_SALES
    """
    # --- Determine ECR lookback window ---
    forecast_month_bounds = (
        session.table(TFT_FORECAST_TABLE)
        .select(
            F.min("MONTH_OF_SALE").alias("MIN_FORECAST_MONTH"),
            F.max("MONTH_OF_SALE").alias("MAX_FORECAST_MONTH")
        )
        .collect()[0]
    )
    earliest_forecast_month = forecast_month_bounds["MIN_FORECAST_MONTH"]
    latest_forecast_month   = forecast_month_bounds["MAX_FORECAST_MONTH"]
    lookback_start = F.add_months(F.lit(earliest_forecast_month), -LOOKBACK_MONTHS)

    logger.info("ECR lookback: %s months before %s", LOOKBACK_MONTHS, earliest_forecast_month)

    # --- Load ECR for the lookback window ---
    ecr_raw = (
        session.table(ECR_TABLE)
        .filter(F.col("X_CUSTOMER_TYPE").isin(list(CUSTOMER_TYPE)))
        .filter(F.col("CAL_DATE") >= lookback_start)
        .filter(F.col("CAL_DATE") <  F.lit(latest_forecast_month))
        .with_column("CAL_DATE", F.to_date(F.col("CAL_DATE")))
        .with_column(
            "NET_SALES",
            F.col("INVOICED_SALES") + F.col("CANCELLED_SALES") + F.col("RETURNED_SALES")
        )
    )

    # Attach family codes + parent dealer
    sku_ref = (
        session.table(SKU_SUPERCEDENCE_TABLE).drop("UPDATED_ON")
    )
    for old_col in sku_ref.columns:
        sku_ref = sku_ref.rename(old_col, old_col.replace('"', ''))
    sku_ref = sku_ref.with_column_renamed("UNIQUE FAMILY CODE", "UNIQUE_FAMILY_CODE") \
                     .select("SKU", "MODEL", "UNIQUE_FAMILY_CODE")

    parent_map = (
        session.table(PARENT_DEALER_VIEW)
        .filter(~F.col("X_DEALER_CODE_HIER").is_null())
        .select(
            F.col("X_DEALER_CODE_HIER").alias("DEALER_CODE"),
            F.trim(F.split(F.col("PAR_ORG_NAME"), F.lit('-'))[0]).alias("PARENT_DEALER_CODE")
        ).distinct()
    )

    ecr = (
        ecr_raw
        .join(sku_ref,    on=["SKU", "MODEL"], how="left")
        .join(parent_map, on="DEALER_CODE",    how="left")
    )

    # Remap to current OBD SKU if enabled
    if IS_OBD:
        obd = (
            session.table(OBD_MAPPING_TABLE)
            .select(
                F.col("PREVIOUS_OBD_SKU").alias("SKU"),
                F.col("CURRENT_OBD_SKU")
            )
        )
        ecr = (
            ecr
            .join(obd, on="SKU", how="left")
            .with_column("SKU", F.coalesce(F.col("CURRENT_OBD_SKU"), F.col("SKU")))
            .drop("CURRENT_OBD_SKU")
        )

    # --- Cross-join ECR with forecast months, then window to each month's lookback ---
    forecast_months = forecast.select("MONTH_OF_SALE").distinct()
    ecr_with_month  = ecr.join(forecast_months, how="cross")

    ecr_windowed = ecr_with_month.filter(
        (F.col("CAL_DATE") >= F.add_months(F.col("MONTH_OF_SALE"), F.lit(-LOOKBACK_MONTHS))) &
        (F.col("CAL_DATE") <  F.col("MONTH_OF_SALE"))
    )

    # Aggregate SKU sales within the lookback window for each forecast month
    dealer_family_sku_sales = (
        ecr_windowed
        .group_by(["MONTH_OF_SALE", "PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU"])
        .agg(F.sum("NET_SALES").alias("DEALER_SKU_SALES"))
    )

    logger.info("Dealer-SKU-Family sales computed for lookback window.")

    # --- Expand forecast to one row per active SKU ---
    forecast_sku = (
        forecast
        .join(active_skus.select("UNIQUE_FAMILY_CODE", "SKU"),
              on="UNIQUE_FAMILY_CODE", how="left")
        .join(num_active, on="UNIQUE_FAMILY_CODE", how="left")
    )

    # --- Attach historical SKU sales ---
    disaggregated = (
        forecast_sku
        .join(
            dealer_family_sku_sales.select(
                "MONTH_OF_SALE", "PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE",
                "SKU", "DEALER_SKU_SALES"
            ),
            on=["MONTH_OF_SALE", "PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU"],
            how="left"
        )
        .with_column("DEALER_SKU_SALES",
                     F.coalesce(F.col("DEALER_SKU_SALES"), F.lit(0.0)))
    )

    # --- Compute family total (sum of active SKU sales only) ---
    family_window = Window.partition_by(
        "MONTH_OF_SALE", "PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE"
    )
    disaggregated = disaggregated.with_column(
        "FAMILY_TOTAL_SALES",
        F.sum("DEALER_SKU_SALES").over(family_window)
    )

    # --- Compute proportion for each SKU (mirrors percentsku() from Step3.py) ---
    disaggregated = disaggregated.with_column(
        "PERCENT_PROPORTION",
        F.when(
            F.col("FAMILY_TOTAL_SALES") == F.lit(0),
            F.lit(1.0) / F.col("NUM_ACTIVE_SKUS")        # no history -> equal split
        ).when(
            F.col("DEALER_SKU_SALES") == F.lit(0),
            F.lit(0.0)                                    # family has history, SKU doesn't -> 0
        ).otherwise(
            F.col("DEALER_SKU_SALES") / F.col("FAMILY_TOTAL_SALES")
        )
    )

    # --- Apply proportion to get SKU-level predicted sales ---
    disaggregated = disaggregated.with_column(
        "PREDICTED_SALES_SKU_TFT",
        F.when(
            F.col("NUM_ACTIVE_SKUS") == F.lit(1),
            F.col("PREDICTED_SALES_TFT")                  # single SKU -> no split needed
        ).otherwise(
            F.col("PERCENT_PROPORTION") * F.col("PREDICTED_SALES_TFT")
        )
    )

    # --- Flag rows with no historical sales (useful for downstream review) ---
    disaggregated = disaggregated.with_column(
        "NO_HISTORY_FLAG",
        F.when(F.col("FAMILY_TOTAL_SALES") == F.lit(0), F.lit(True))
         .otherwise(F.lit(False))
    )

    logger.info("Disaggregation complete.")
    return disaggregated


# =============================================================================
# BLOCK F — TRANSIT TIME JOIN
# =============================================================================

def join_transit_data(session, soq_df, logger):
    """
    Joins the disaggregated SOQ data with max/min/avg transit times
    per PARENT_DEALER_CODE + SKU from the pre-built transit view.
    """
    transit = session.table(TRANSIT_TABLE)
    result  = soq_df.join(transit, on=["PARENT_DEALER_CODE", "SKU"], how="left")
    logger.info("Transit data joined. Output rows: %s", result.count())
    return result


# =============================================================================
# BLOCK G — STOCK DATA MAPPING
# =============================================================================

def map_stock_to_forecast(session, forecast_df, planning_month, stock_period,
                           active_skus, num_active, logger):
    """
    Attaches stock-on-date to each forecast row.
    Steps:
      1. Compute the stock snapshot date from planning_month + stock_period
      2. Load stock, aggregate to parent-dealer level
      3. Expand forecast rows to one per active SKU
      4. Join stock data and filter out rows with no active SKU count
    """
    stock_date = get_stock_date(stock_period, planning_month)
    logger.info("Stock date for period '%s': %s", stock_period, stock_date)

    stock_raw = load_stock_data(session, stock_date, logger)

    # Aggregate stock to parent-dealer + SKU level
    parent_map = (
        session.table(PARENT_DEALER_VIEW)
        .filter(~F.col("X_DEALER_CODE_HIER").is_null())
        .select(
            F.col("X_DEALER_CODE_HIER").alias("DEALER_CODE"),
            F.trim(F.split(F.col("PAR_ORG_NAME"), F.lit('-'))[0]).alias("PARENT_DEALER_CODE")
        ).distinct()
    )

    sku_ref = (
        session.table(SKU_SUPERCEDENCE_TABLE).drop("UPDATED_ON")
    )
    for old_col in sku_ref.columns:
        sku_ref = sku_ref.rename(old_col, old_col.replace('"', ''))
    sku_ref = sku_ref.with_column_renamed("UNIQUE FAMILY CODE", "UNIQUE_FAMILY_CODE") \
                     .select("SKU", "MODEL", "UNIQUE_FAMILY_CODE")

    stock = (
        stock_raw
        .join(parent_map, on="DEALER_CODE", how="left")
        .join(sku_ref,    on=["SKU", "MODEL"], how="left")
        .group_by(["PARENT_DEALER_CODE", "SKU"])
        .agg(F.sum("STK_AS_ON_DATE").alias("STK_AS_ON_DATE"))
    )

    # Expand forecast to one row per active SKU and attach stock
    soq_sku = (
        forecast_df
        .join(num_active,  on="UNIQUE_FAMILY_CODE", how="left")
        .join(active_skus.select("UNIQUE_FAMILY_CODE", "SKU"),
              on="UNIQUE_FAMILY_CODE", how="left")
        .join(stock, on=["PARENT_DEALER_CODE", "SKU"], how="left")
        .filter(~F.col("NUM_ACTIVE_SKUS").is_null())  # drop families with no active SKUs
    )

    logger.info("Stock mapping complete. Output rows: %s", soq_sku.count())
    return soq_sku, stock_date


# =============================================================================
# BLOCK H — DEMAND VARIABILITY
# Measures how erratic each dealer-SKU or dealer-family series is
# Standard deviation of month-over-month deltas over trailing 12 months
# =============================================================================

def compute_demand_variability(session, planning_month, run_date, run_version, logger):
    """
    Calculates demand variability at SKU level and saves to DEMAND_VARIABILITY_TABLE.
    Variability = std dev of first differences (month-over-month change) in NET_SALES.
    Uses a 12-month lookback from run_date.
    """
    from dateutil.relativedelta import relativedelta

    end_date   = datetime.datetime.strptime(run_date, "%Y%m%d").replace(day=1)
    start_date = (end_date - relativedelta(months=12)).strftime("%Y-%m-%d")
    end_date   = end_date.strftime("%Y-%m-%d")

    ecr = load_ecr_data(session, start_date, end_date, logger)

    # Monthly aggregation at dealer + family + SKU level
    monthly = (
        ecr
        .with_column("CAL_MONTH", F.date_trunc("MONTH", F.col("CAL_DATE")))
        .group_by(["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU", "CAL_MONTH"])
        .agg(F.sum("NET_SALES").alias("NET_SALES"))
    )

    # Compute first difference (current month - previous month) using lag window
    sku_window = Window.partition_by(
        "PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU"
    ).order_by("CAL_MONTH")

    monthly = monthly.with_column(
        "PREV_NET_SALES", F.lag("NET_SALES").over(sku_window)
    ).with_column(
        "DELTA", F.col("NET_SALES") - F.col("PREV_NET_SALES")
    ).filter(F.col("PREV_NET_SALES").is_not_null())   # drop first row per series (no lag)

    # Std dev of deltas per series = demand variability
    demand_variability = (
        monthly
        .group_by(["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU"])
        .agg(F.stddev("DELTA").alias("DEMAND_VARIABILITY"))
        # If a series has fewer than 2 deltas stddev is null -> default to 1
        .with_column("DEMAND_VARIABILITY",
                     F.coalesce(F.col("DEMAND_VARIABILITY"), F.lit(1.0)))
        .with_column("PLANNING_MONTH",  F.lit(planning_month))
        .with_column("ECR_START_DATE",  F.lit(start_date))
        .with_column("ECR_END_DATE",    F.lit(end_date))
        .with_column("RUN_DATE",        F.lit(run_date))
        .with_column("RUN_VERSION",     F.lit(run_version))
        .with_column("IS_OBD",          F.lit(obd_flag))
    )

    demand_variability.write.mode("append").save_as_table(DEMAND_VARIABILITY_TABLE)
    logger.info("Demand variability (SKU level) written to %s", DEMAND_VARIABILITY_TABLE)


def compute_demand_variability_by_family(session, planning_month, run_date, run_version, logger):
    """
    Same as compute_demand_variability but aggregated at family level (no SKU).
    Saves to DEMAND_VARIABILITY_FAMILY_TABLE.
    """
    from dateutil.relativedelta import relativedelta

    end_date   = datetime.datetime.strptime(run_date, "%Y%m%d").replace(day=1)
    start_date = (end_date - relativedelta(months=12)).strftime("%Y-%m-%d")
    end_date   = end_date.strftime("%Y-%m-%d")

    ecr = load_ecr_data(session, start_date, end_date, logger)

    monthly = (
        ecr
        .with_column("CAL_MONTH", F.date_trunc("MONTH", F.col("CAL_DATE")))
        .group_by(["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "CAL_MONTH"])
        .agg(F.sum("NET_SALES").alias("NET_SALES"))
    )

    family_window = Window.partition_by(
        "PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE"
    ).order_by("CAL_MONTH")

    monthly = monthly.with_column(
        "PREV_NET_SALES", F.lag("NET_SALES").over(family_window)
    ).with_column(
        "DELTA", F.col("NET_SALES") - F.col("PREV_NET_SALES")
    ).filter(F.col("PREV_NET_SALES").is_not_null())

    demand_variability = (
        monthly
        .group_by(["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE"])
        .agg(F.stddev("DELTA").alias("DEMAND_VARIABILITY"))
        .with_column("DEMAND_VARIABILITY",
                     F.coalesce(F.col("DEMAND_VARIABILITY"), F.lit(1.0)))
        .with_column("PLANNING_MONTH",  F.lit(planning_month))
        .with_column("ECR_START_DATE",  F.lit(start_date))
        .with_column("ECR_END_DATE",    F.lit(end_date))
        .with_column("RUN_DATE",        F.lit(run_date))
        .with_column("RUN_VERSION",     F.lit(run_version))
        .with_column("IS_OBD",          F.lit(obd_flag))
    )

    demand_variability.write.mode("append").save_as_table(DEMAND_VARIABILITY_FAMILY_TABLE)
    logger.info("Demand variability (family level) written to %s", DEMAND_VARIABILITY_FAMILY_TABLE)


# =============================================================================
# BLOCK I — VALIDATION
# =============================================================================

def validate_disaggregation(output, logger):
    """
    Sanity check: for rows that have historical data (NO_HISTORY_FLAG=False),
    re-summing SKU-level predictions per family+month must equal the original
    family-level forecast within a floating-point tolerance.
    Also logs: max absolute difference, rows with no history, total output rows.
    """
    check = (
        output.filter(F.col("NO_HISTORY_FLAG") == F.lit(False))
        .group_by(["MONTH_OF_SALE", "PARENT_DEALER_CODE_MODEL_FAMILY"])
        .agg(
            F.max("PREDICTED_SALES_TFT").alias("ORIGINAL"),
            F.sum("PREDICTED_SALES_SKU_TFT").alias("REAGGREGATED")
        )
        .with_column("DIFF", F.abs(F.col("ORIGINAL") - F.col("REAGGREGATED")))
    )

    max_diff         = check.agg(F.max("DIFF").alias("MAX_DIFF")).collect()[0]["MAX_DIFF"]
    no_history_count = output.filter(F.col("NO_HISTORY_FLAG") == F.lit(True)).count()
    total_rows       = output.count()

    logger.info("Validation | MAX re-aggregation diff: %s | No-history rows: %s | Total rows: %s",
                max_diff, no_history_count, total_rows)
    return max_diff, no_history_count, total_rows


# =============================================================================
# MAIN
# =============================================================================

def main(session: snowpark.Session):
    logger, log_stream = _setup_logger()
    logger.info("========== SOQ Step 3 Pipeline Started ==========")

    try:
        # --- Load shared reference data once (reused across all planning months) ---
        active_skus, num_active = load_sku_supercedence(session, logger)

        # -----------------------------------------------------------------------
        # OUTER LOOP: one iteration per planning month
        # -----------------------------------------------------------------------
        for planning_month in MONTHS:
            logger.info("---------- Processing planning month: %s ----------", planning_month)

            # STEP 1: Parse and store the TFT forecast for this month
            forecast = process_forecast_table(session, planning_month, logger)

            # STEP 2: Disaggregate family forecast to SKU level
            disaggregated = disaggregate_forecast(
                session, forecast, active_skus, num_active, planning_month, logger
            )

            # STEP 3: Select final output columns and round
            output = disaggregated.select(
                "MONTH_OF_SALE",
                "PARENT_DEALER_CODE_MODEL_FAMILY",
                "PARENT_DEALER_CODE",
                "UNIQUE_FAMILY_CODE",
                "SKU",
                "NUM_ACTIVE_SKUS",
                "PREDICTED_SALES_TFT",
                F.round("PERCENT_PROPORTION",    5).alias("PERCENT_PROPORTION"),
                F.round("PREDICTED_SALES_SKU_TFT", 4).alias("PREDICTED_SALES_SKU_TFT"),
                "DEALER_SKU_SALES",
                "FAMILY_TOTAL_SALES",
                "NO_HISTORY_FLAG"
            )

            # STEP 4: Validate re-aggregation integrity
            validate_disaggregation(output, logger)

            # STEP 5: Save disaggregated SKU-level output
            output.write.mode("overwrite").save_as_table(OUTPUT_TABLE)
            logger.info("Disaggregated output saved to %s", OUTPUT_TABLE)

            # ------------------------------------------------------------------
            # INNER LOOP: one iteration per stock date type (e.g. "first", "end")
            # ------------------------------------------------------------------
            for stock_period in STOCK_DATE_TYPE:
                logger.info("--- Stock period: %s ---", stock_period)

                # STEP 6: Attach stock data + expand to active SKU rows
                soq_data_sku, stock_date = map_stock_to_forecast(
                    session, forecast, planning_month, stock_period,
                    active_skus, num_active, logger
                )

                # STEP 7: ECR aggregation for last 3 months (for proportion calc in SOQ)
                from dateutil.relativedelta import relativedelta
                run_dt     = datetime.datetime.strptime(RUN_DATE, '%Y%m%d').replace(day=1)
                ecr_start  = (run_dt - relativedelta(months=3)).strftime("%Y-%m-%d")
                ecr_end    = run_dt.strftime("%Y-%m-%d")
                ecr_data   = load_ecr_data(session, ecr_start, ecr_end, logger)

                # Family-level and SKU-level sales aggregates from ECR
                dealer_family_sales = (
                    ecr_data
                    .group_by(["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE"])
                    .agg(F.sum("NET_SALES").alias("DEALER_FAMILY_CODE_NET_SALES"))
                )
                dealer_family_sku_sales = (
                    ecr_data
                    .group_by(["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU"])
                    .agg(F.sum("NET_SALES").alias("DEALER_SKU_SALES"))
                )

                # Attach family-level sales to the SOQ data
                soq_data_sku = (
                    soq_data_sku
                    .join(dealer_family_sales,
                          on=["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE"], how="left")
                    .with_column("DEALER_FAMILY_CODE_NET_SALES",
                                 F.coalesce(F.col("DEALER_FAMILY_CODE_NET_SALES"), F.lit(0.0)))
                    .join(dealer_family_sku_sales,
                          on=["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU"], how="left")
                )

                # Compute sum of active-SKU sales per family (denominator for proportion)
                active_family_window = Window.partition_by(
                    "PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE"
                )
                soq_data_sku = soq_data_sku.with_column(
                    "TOTAL_DEALER_ACTIVE_SKU_SALES",
                    F.sum(
                        F.coalesce(F.col("DEALER_SKU_SALES"), F.lit(0.0))
                    ).over(active_family_window)
                ).with_column(
                    "DEALER_SKU_SALES",
                    F.coalesce(F.col("DEALER_SKU_SALES"), F.lit(0.0))
                )

                # Apply proportion logic (mirrors percentsku from Step3.py)
                soq_data_sku = soq_data_sku.with_column(
                    "PERCENT_PROPORTION",
                    F.round(
                        F.when(
                            F.col("TOTAL_DEALER_ACTIVE_SKU_SALES") == F.lit(0),
                            F.lit(1.0) / F.col("NUM_ACTIVE_SKUS")
                        ).when(
                            F.col("DEALER_SKU_SALES") == F.lit(0),
                            F.lit(0.0)
                        ).otherwise(
                            F.col("DEALER_SKU_SALES") / F.col("TOTAL_DEALER_ACTIVE_SKU_SALES")
                        ),
                        5
                    )
                )

                soq_data_sku = soq_data_sku.with_column(
                    "PREDICTED_SALES_SKU",
                    F.when(
                        F.col("NUM_ACTIVE_SKUS") == F.lit(1),
                        F.col("PREDICTED_SALES_TFT")
                    ).otherwise(
                        F.col("PERCENT_PROPORTION") * F.col("PREDICTED_SALES_TFT")
                    )
                )

                # STEP 8: Join transit lead times
                soq_final = join_transit_data(session, soq_data_sku, logger)

                # STEP 9: Attach metadata and save
                soq_final = (
                    soq_final
                    .drop("DEALER_ACTIVE_SKU_TOTAL_SALES", "TOTAL_DEALER_ACTIVE_SKU_SALES")
                    .drop_duplicates()
                    .with_column("STOCK_DATE_PERIOD", F.lit(stock_period))
                    .with_column("STOCK_DATE",        F.lit(stock_date))
                    .with_column("PLANNING_MONTH",    F.lit(planning_month))
                    .with_column("RUN_DATE",          F.lit(RUN_DATE))
                    .with_column("RUN_VERSION",       F.lit(RUN_VERSION))
                    .with_column("IS_OBD",            F.lit(obd_flag))
                )

                soq_final.write.mode("append").save_as_table(BASE_SOQ_TABLE)
                logger.info("SOQ base table updated: %s | Period: %s",
                            BASE_SOQ_TABLE, stock_period)

        # -----------------------------------------------------------------------
        # Demand variability — computed after all forecast months are processed
        # -----------------------------------------------------------------------
        for planning_month in MONTHS:
            compute_demand_variability(
                session, planning_month, RUN_DATE, RUN_VERSION, logger
            )
            compute_demand_variability_by_family(
                session, planning_month, RUN_DATE, RUN_VERSION, logger
            )

        logger.info("========== SOQ Step 3 Pipeline Complete ==========")

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        raise

    finally:
        print(log_stream.getvalue())

    return session.table(DEMAND_VARIABILITY_FAMILY_TABLE)
