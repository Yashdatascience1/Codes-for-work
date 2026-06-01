import snowflake.snowpark as snowpark
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window
from dateutil.relativedelta import relativedelta
import datetime

# =============================================================================
# CONFIGURATION
# All knobs are here. Nothing is hardcoded below.
# =============================================================================

# The TFT model output: family-level monthly predictions per dealer
TFT_FORECAST_TABLE = 'MOP_DATABASE.SOQ.PREDICTIONS_BY_TFT_JAN_26_TO_APR_26'

# Where we save: SKU-level disaggregated predictions
OUTPUT_TABLE = 'MOP_DATABASE.SOQ.DEALER_SKU_DISAGGREGATION_RESULTS'

# Where we save: full SOQ base table (with stock, transit, metadata)
BASE_SOQ_TABLE = 'MOP_DATABASE.SOQ.SOQ_BASE_TABLE_FINAL_CONCATENATED'

# Where we save: demand variability metrics
DEMAND_VARIABILITY_TABLE        = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_FINAL_VERSION'
DEMAND_VARIABILITY_FAMILY_TABLE = 'MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_MODEL_FAMILY_FINAL_VERSION'

# Where forecasts parsed from TFT are staged before further processing
PREDICTION_TABLE = 'MOP_DATABASE.SOQ.SOQ_PREDICTION_FINAL_VERSION'

# Source tables
ECR_TABLE      = 'ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS'
STOCK_TABLE    = 'ANALYTICS_DATABASE.ANALYTICS_SALES.STOCK_AVAILABILITY'
SKU_SUPERCEDENCE_TABLE = 'MOP_DATABASE.SOQ.SKU_SUPERCEDENCE_MODEL_FAMILY_MAY_2026_UPDATED_V2'
OBD_MAPPING_TABLE      = 'MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW'
PARENT_DEALER_VIEW     = 'FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH'
TRANSIT_TABLE          = 'MOP_DATABASE.SOQ.PARENT_DEALER_TRANSIT_TIME_SKU_NEW'

# Run parameters
CUSTOMER_TYPE   = ('Individual',)  # only retail customers
IS_OBD          = True             # remap old SKU variants to current OBD SKU
LOOKBACK_MONTHS = 3                # months of history used to compute SKU proportions
MID_DATE        = 15               # day-of-month used for "mid" stock snapshot
RUN_DATE        = datetime.datetime.today().strftime('%Y%m%d')
RUN_VERSION     = 33
MONTHS          = ['2026-06-01']   # planning months to process
STOCK_DATE_TYPE = ['first']        # stock snapshot types: "first", "end", or "mid"

obd_flag = 'Y' if IS_OBD else 'N'


def main(session: snowpark.Session):

    # =========================================================================
    # BLOCK 1 — LOAD THE SKU SUPERCEDENCE TABLE
    #
    # This table is the product master. It tells us:
    #   - Which SKUs are "active" (currently sold)
    #   - Which UNIQUE_FAMILY_CODE each SKU belongs to
    #     (e.g. "ACTIVA<>DRUM<>SELF<>ALLOY<>RED")
    #
    # We use it in two ways:
    #   a) active_skus  — to expand each family forecast into one row per active SKU
    #   b) num_active   — to know how many SKUs to split equally when history is zero
    # =========================================================================

    sku_raw = session.table(SKU_SUPERCEDENCE_TABLE)

    # UPDATED_ON is stored as an integer in YYYYMMDD format (e.g. 20260101).
    # Convert it to a real date so we can read the latest update date properly.
    sku_raw = sku_raw.with_column(
        "UPDATED_ON",
        F.to_date(F.col("UPDATED_ON").cast("string"), F.lit("YYYYMMDD"))
    )

    # Note the latest date for auditability, then drop the column — we don't
    # need it downstream, and it causes join ambiguity if left in.
    latest_sku_update = sku_raw.select(F.max("UPDATED_ON")).collect()[0][0]
    print(f"SKU supercedence last updated on: {latest_sku_update}")
    sku_raw = sku_raw.drop("UPDATED_ON")

    # Snowpark sometimes wraps column names in double-quotes after joins/loads.
    # Strip them so all col() references below work without escaping.
    for old_col in sku_raw.columns:
        sku_raw = sku_raw.rename(old_col, old_col.replace('"', ''))

    # The column is named "UNIQUE FAMILY CODE" (with spaces) in the source.
    # Rename to UNIQUE_FAMILY_CODE (underscores) so it joins cleanly.
    sku_raw = sku_raw.with_column_renamed("UNIQUE FAMILY CODE", "UNIQUE_FAMILY_CODE")

    # Keep only active SKUs — inactive/discontinued SKUs should not receive any
    # forecast allocation. This is a critical business rule.
    active_skus = sku_raw.filter(F.col("SKUSTATUS") == F.lit("active"))

    # For each family, count how many active SKUs exist.
    # This count is the fallback denominator when a dealer has zero historical
    # sales for that family (we split equally across all active SKUs).
    num_active = (
        active_skus
        .group_by("UNIQUE_FAMILY_CODE")
        .agg(F.count_distinct("SKU").alias("NUM_ACTIVE_SKUS"))
    )

    print(f"Unique family codes with active SKUs: "
          f"{num_active.select(F.count_distinct('UNIQUE_FAMILY_CODE')).collect()[0][0]}")


    # =========================================================================
    # BLOCK 2 — BUILD THE PARENT DEALER MAPPING
    #
    # Dealers are organised in a hierarchy. Individual DEALER_CODEs roll up to
    # a PARENT_DEALER_CODE (a regional group). All forecasting and SOQ
    # calculations happen at the PARENT_DEALER_CODE level.
    #
    # PAR_ORG_NAME looks like "DELHI-NORTH" — we split on "-" and take the
    # first token to get "DELHI" as the PARENT_DEALER_CODE.
    # =========================================================================

    parent_dealer_mapping = (
        session.table(PARENT_DEALER_VIEW)
        .filter(~F.col("X_DEALER_CODE_HIER").is_null())
        .select(
            F.col("X_DEALER_CODE_HIER").alias("DEALER_CODE"),
            F.trim(F.split(F.col("PAR_ORG_NAME"), F.lit('-'))[0]).alias("PARENT_DEALER_CODE")
        )
        .distinct()   # one row per dealer; hierarchy table can have duplicates
    )

    print(f"Unique dealer codes in parent mapping: {parent_dealer_mapping.count()}")


    # =========================================================================
    # BLOCK 3 — OUTER LOOP: PROCESS EACH PLANNING MONTH
    #
    # MONTHS is a list (usually just one entry) of planning months.
    # Each month is processed independently through the full pipeline.
    # =========================================================================

    for planning_month in MONTHS:
        print(f"\n========== Planning month: {planning_month} ==========")


        # =====================================================================
        # BLOCK 3a — PARSE THE TFT FORECAST TABLE
        #
        # The TFT model outputs predictions at the FAMILY level per dealer per
        # month. The dealer+family identity is stored in a single composite key:
        #   PARENT_DEALER_CODE_MODEL_FAMILY = "DEALER<>MODEL_FAMILY<>ATTR..."
        #
        # We need to split this into separate columns:
        #   PARENT_DEALER_CODE  — the part before the first '<>'
        #   UNIQUE_FAMILY_CODE  — everything after the first '<>'
        # =====================================================================

        forecast = (
            session.table(TFT_FORECAST_TABLE)
            .with_column("MONTH_OF_SALE", F.to_date(F.col("MONTH_OF_SALE")))
            .filter(F.col("MONTH_OF_SALE") == F.lit(planning_month))
        )

        # Extract the dealer code: split on '<>' and take element [0]
        forecast = forecast.with_column(
            "PARENT_DEALER_CODE",
            F.trim(F.split(F.col("PARENT_DEALER_CODE_MODEL_FAMILY"), F.lit('<>'))[0])
        )

        # Extract the family code: substring from position after the first '<>'
        # charindex returns the position of '<>' in the string.
        # Adding 2 skips past the two-character '<>' delimiter itself.
        forecast = forecast.with_column(
            "UNIQUE_FAMILY_CODE",
            F.substr(
                F.col("PARENT_DEALER_CODE_MODEL_FAMILY"),
                F.charindex(F.lit('<>'), F.col("PARENT_DEALER_CODE_MODEL_FAMILY")) + F.lit(2)
            )
        )

        # Persist the parsed forecast for auditability / downstream use
        forecast.write.mode("append").save_as_table(PREDICTION_TABLE)

        print(f"Forecast rows for {planning_month}: {forecast.count()}")
        print(f"Unique dealers in forecast: "
              f"{forecast.select(F.count_distinct('PARENT_DEALER_CODE')).collect()[0][0]}")


        # =====================================================================
        # BLOCK 3b — COMPUTE SKU PROPORTIONS FROM HISTORICAL ECR SALES
        #
        # The TFT model forecasts at the family level (e.g. total ACTIVA sales
        # for dealer DELHI in June). We need to split that number across
        # individual SKUs (e.g. ACTIVA DRUM RED, ACTIVA DRUM BLUE, etc.).
        #
        # The split is based on each SKU's share of the family's total sales
        # over the last LOOKBACK_MONTHS months.
        #
        # For each forecast month, we look back LOOKBACK_MONTHS months from
        # that forecast date to compute the shares. This is done via a
        # cross-join + filter pattern (see below).
        # =====================================================================

        # Determine the ECR date window (lookback before the earliest forecast month)
        bounds = (
            session.table(TFT_FORECAST_TABLE)
            .select(
                F.min("MONTH_OF_SALE").alias("MIN_FORECAST_MONTH"),
                F.max("MONTH_OF_SALE").alias("MAX_FORECAST_MONTH")
            )
            .collect()[0]
        )
        earliest_forecast_month = bounds["MIN_FORECAST_MONTH"]
        latest_forecast_month   = bounds["MAX_FORECAST_MONTH"]

        # Pull ECR from LOOKBACK_MONTHS before the earliest forecast month
        # up to (but not including) the latest forecast month.
        # We need more than just the lookback for one month because we will
        # roll this window per-month using the cross-join below.
        lookback_start = F.add_months(F.lit(earliest_forecast_month), -LOOKBACK_MONTHS)

        ecr_raw = (
            session.table(ECR_TABLE)
            .filter(F.col("X_CUSTOMER_TYPE").isin(list(CUSTOMER_TYPE)))
            .filter(F.col("CAL_DATE") >= lookback_start)
            .filter(F.col("CAL_DATE") <  F.lit(latest_forecast_month))
            .with_column("CAL_DATE", F.to_date(F.col("CAL_DATE")))
            .with_column(
                "NET_SALES",
                F.col("INVOICED_SALES") + F.col("CANCELLED_SALES") + F.col("RETURNED_SALES")
                # Note: CANCELLED and RETURNED are stored as negative values,
                # so this addition correctly computes NET_SALES.
            )
        )

        # Attach UNIQUE_FAMILY_CODE from the SKU supercedence table.
        # Left join: ECR rows for SKUs not in the supercedence table keep all
        # their data but get a NULL family code — they won't match any forecast
        # row later and will effectively be ignored.
        sku_ref = active_skus.select("SKU", "MODEL", "UNIQUE_FAMILY_CODE")
        ecr_raw = ecr_raw.join(sku_ref, on=["SKU", "MODEL"], how="left")

        # Attach PARENT_DEALER_CODE from the dealer hierarchy table.
        ecr_raw = ecr_raw.join(parent_dealer_mapping, on="DEALER_CODE", how="left")

        # If IS_OBD is True, remap each sold SKU to its current OBD equivalent.
        # This ensures historical sales of a discontinued SKU are credited to
        # the new SKU that replaced it, giving it the correct sales history.
        if IS_OBD:
            obd = (
                session.table(OBD_MAPPING_TABLE)
                .select(
                    F.col("PREVIOUS_OBD_SKU").alias("SKU"),
                    F.col("CURRENT_OBD_SKU")
                )
            )
            ecr_raw = (
                ecr_raw
                .join(obd, on="SKU", how="left")
                .with_column("SKU", F.coalesce(F.col("CURRENT_OBD_SKU"), F.col("SKU")))
                .drop("CURRENT_OBD_SKU")
            )

        # Cross-join ECR with the distinct forecast months, then filter to
        # each month's specific lookback window.
        #
        # Why cross-join + filter instead of a simpler date range?
        # Because each forecast month has its OWN lookback window:
        #   - June 2026 forecast uses March/April/May 2026 sales
        #   - July 2026 forecast uses April/May/June 2026 sales
        # The cross-join gives every ECR row a copy for every forecast month,
        # and the filter then keeps only ECR rows within that month's window.
        forecast_months = forecast.select("MONTH_OF_SALE").distinct()
        ecr_with_month  = ecr_raw.join(forecast_months, how="cross")

        ecr_windowed = ecr_with_month.filter(
            (F.col("CAL_DATE") >= F.add_months(F.col("MONTH_OF_SALE"), F.lit(-LOOKBACK_MONTHS))) &
            (F.col("CAL_DATE") <  F.col("MONTH_OF_SALE"))
        )

        # Aggregate: total sales per dealer + family + SKU for each forecast month's window
        dealer_family_sku_sales = (
            ecr_windowed
            .group_by(["MONTH_OF_SALE", "PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU"])
            .agg(F.sum("NET_SALES").alias("DEALER_SKU_SALES"))
        )


        # =====================================================================
        # BLOCK 3c — EXPAND FORECAST TO ONE ROW PER ACTIVE SKU
        #
        # The forecast table has one row per (dealer, family, month).
        # We join it with active_skus to get one row per (dealer, family, SKU, month).
        # Then we join num_active to know how many SKUs are in that family.
        # =====================================================================

        forecast_sku = (
            forecast
            .join(active_skus.select("UNIQUE_FAMILY_CODE", "SKU"),
                  on="UNIQUE_FAMILY_CODE", how="left")
            # left join: families with no active SKUs get a NULL SKU row
            .join(num_active, on="UNIQUE_FAMILY_CODE", how="left")
        )


        # =====================================================================
        # BLOCK 3d — ATTACH HISTORICAL SKU SALES TO EACH FORECAST ROW
        #
        # Left join the dealer-SKU sales history onto the expanded forecast.
        # Rows with no history (new dealer-SKU combos) get DEALER_SKU_SALES = 0.
        # =====================================================================

        disaggregated = (
            forecast_sku
            .join(
                dealer_family_sku_sales.select(
                    "MONTH_OF_SALE", "PARENT_DEALER_CODE",
                    "UNIQUE_FAMILY_CODE", "SKU", "DEALER_SKU_SALES"
                ),
                on=["MONTH_OF_SALE", "PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU"],
                how="left"
            )
            .with_column("DEALER_SKU_SALES",
                         F.coalesce(F.col("DEALER_SKU_SALES"), F.lit(0.0)))
        )


        # =====================================================================
        # BLOCK 3e — COMPUTE FAMILY TOTAL SALES (DENOMINATOR FOR PROPORTION)
        #
        # For each (dealer, family, month), sum the DEALER_SKU_SALES across
        # all active SKUs. This is the denominator used to compute each SKU's
        # share. Because we only expanded to ACTIVE SKUs, this total only counts
        # active SKU sales — inactive SKU sales are excluded.
        #
        # We use a window function (not a group_by + join) so that each row
        # keeps its SKU-level detail while also having the family total.
        # =====================================================================

        family_window = Window.partition_by(
            "MONTH_OF_SALE", "PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE"
        )

        disaggregated = disaggregated.with_column(
            "FAMILY_TOTAL_SALES",
            F.sum("DEALER_SKU_SALES").over(family_window)
        )


        # =====================================================================
        # BLOCK 3f — COMPUTE PERCENT_PROPORTION FOR EACH SKU
        #
        # Three cases (this mirrors the percentsku() function in Step3.py):
        #
        #   Case 1 — FAMILY_TOTAL_SALES == 0 (no history for this family at this dealer):
        #            Split the forecast equally across all active SKUs.
        #            Proportion = 1 / NUM_ACTIVE_SKUS
        #
        #   Case 2 — FAMILY_TOTAL_SALES > 0 but DEALER_SKU_SALES == 0 (family sold
        #            products historically but not this specific SKU):
        #            This SKU gets nothing. Proportion = 0.
        #
        #   Case 3 — Both > 0 (normal case):
        #            Proportion = this SKU's sales / total family sales
        # =====================================================================

        disaggregated = disaggregated.with_column(
            "PERCENT_PROPORTION",
            F.when(
                F.col("FAMILY_TOTAL_SALES") == F.lit(0),
                F.lit(1.0) / F.col("NUM_ACTIVE_SKUS")     # Case 1: equal split
            ).when(
                F.col("DEALER_SKU_SALES") == F.lit(0),
                F.lit(0.0)                                  # Case 2: no history for this SKU
            ).otherwise(
                F.col("DEALER_SKU_SALES") / F.col("FAMILY_TOTAL_SALES")  # Case 3: proportional
            )
        )


        # =====================================================================
        # BLOCK 3g — APPLY PROPORTION TO GET SKU-LEVEL PREDICTED SALES
        #
        # Special case: if a family has only ONE active SKU, no splitting is
        # needed — the entire family forecast belongs to that SKU.
        # For multiple SKUs, multiply each SKU's proportion by the family total.
        # =====================================================================

        disaggregated = disaggregated.with_column(
            "PREDICTED_SALES_SKU_TFT",
            F.when(
                F.col("NUM_ACTIVE_SKUS") == F.lit(1),
                F.col("PREDICTED_SALES_TFT")               # single SKU: pass through as-is
            ).otherwise(
                F.col("PERCENT_PROPORTION") * F.col("PREDICTED_SALES_TFT")
            )
        )

        # Flag rows where the dealer had zero history for this family.
        # Downstream users can filter on this to see which allocations are
        # estimates (equal split) vs. data-driven (proportional).
        disaggregated = disaggregated.with_column(
            "NO_HISTORY_FLAG",
            F.when(F.col("FAMILY_TOTAL_SALES") == F.lit(0), F.lit(True))
             .otherwise(F.lit(False))
        )


        # =====================================================================
        # BLOCK 3h — SELECT FINAL OUTPUT COLUMNS AND SAVE
        # =====================================================================

        output = disaggregated.select(
            "MONTH_OF_SALE",
            "PARENT_DEALER_CODE_MODEL_FAMILY",
            "PARENT_DEALER_CODE",
            "UNIQUE_FAMILY_CODE",
            "SKU",
            "NUM_ACTIVE_SKUS",
            "PREDICTED_SALES_TFT",
            F.round("PERCENT_PROPORTION",      5).alias("PERCENT_PROPORTION"),
            F.round("PREDICTED_SALES_SKU_TFT", 4).alias("PREDICTED_SALES_SKU_TFT"),
            "DEALER_SKU_SALES",
            "FAMILY_TOTAL_SALES",
            "NO_HISTORY_FLAG"
        )


        # =====================================================================
        # BLOCK 3i — VALIDATION: RE-AGGREGATION CHECK
        #
        # For rows that had historical data (NO_HISTORY_FLAG = False), summing
        # the SKU-level predictions back to the family level should exactly
        # reproduce the original family forecast (PREDICTED_SALES_TFT).
        #
        # If the max difference is non-zero (beyond floating point), something
        # is wrong with the proportion calculation.
        # =====================================================================

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

        print(f"Validation | Max re-aggregation diff: {max_diff} | "
              f"No-history rows: {no_history_count} | Total rows: {total_rows}")

        # Save the disaggregated SKU forecast
        output.write.mode("overwrite").save_as_table(OUTPUT_TABLE)
        print(f"Disaggregated output saved to: {OUTPUT_TABLE}")


        # =====================================================================
        # BLOCK 4 — INNER LOOP: STOCK-BASED SOQ CALCULATION
        #
        # For each stock date type ("first", "end", "mid"), we compute the
        # actual SOQ (Suggested Order Quantity) by combining:
        #   - The SKU-level forecast (how much the dealer will sell)
        #   - Current stock on hand (how much they already have)
        #   - Transit lead time (how long delivery takes)
        # =====================================================================

        for stock_period in STOCK_DATE_TYPE:
            print(f"\n--- Stock period: {stock_period} ---")


            # =================================================================
            # BLOCK 4a — DETERMINE STOCK SNAPSHOT DATE
            #
            # "first" = last day of the month BEFORE planning_month
            #           (opening stock at start of planning month)
            # "end"   = last day of planning_month (closing stock at month end)
            # "mid"   = MID_DATE of the month before planning_month
            # =================================================================

            run_month = datetime.datetime.strptime(planning_month, "%Y-%m-%d") - relativedelta(months=1)
            if stock_period == "end":
                stock_date = (
                    datetime.datetime.strptime(planning_month, "%Y-%m-%d").replace(day=1)
                    - relativedelta(days=1)
                ).strftime("%Y-%m-%d")
            elif stock_period == "first":
                stock_date = (run_month.replace(day=1) - relativedelta(days=1)).strftime("%Y-%m-%d")
            else:  # "mid"
                stock_date = run_month.replace(day=MID_DATE).strftime("%Y-%m-%d")

            print(f"Stock snapshot date: {stock_date}")


            # =================================================================
            # BLOCK 4b — LOAD STOCK DATA
            #
            # Pull CLOSING_STOCK from STOCK_AVAILABILITY as of stock_date.
            # If IS_OBD=True, map old SKU variants to current OBD SKU and sum
            # stock across variants so each current SKU shows total stock.
            # =================================================================

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
                    # Re-aggregate after OBD remap: old + new variant stock sums to one row
                    .group_by(["DEALER_CODE", "MODEL", "SKU"])
                    .agg(F.sum("STK_AS_ON_DATE").alias("STK_AS_ON_DATE"))
                )

            # Aggregate stock to PARENT_DEALER + SKU level
            # (Stock_availability is at individual dealer level; SOQ is at parent level)
            stock_parent = (
                stock_raw
                .join(parent_dealer_mapping, on="DEALER_CODE", how="left")
                .group_by(["PARENT_DEALER_CODE", "SKU"])
                .agg(F.sum("STK_AS_ON_DATE").alias("STK_AS_ON_DATE"))
            )

            print(f"Stock rows loaded: {stock_parent.count()}")


            # =================================================================
            # BLOCK 4c — EXPAND FORECAST TO ACTIVE SKUs AND ATTACH STOCK
            #
            # Same expansion logic as Block 3c, but this time we also attach
            # the physical stock quantity so SOQ can be computed later.
            # We filter out families that have no active SKUs (num_active null).
            # =================================================================

            soq_data_sku = (
                forecast
                .join(num_active, on="UNIQUE_FAMILY_CODE", how="left")
                .join(active_skus.select("UNIQUE_FAMILY_CODE", "SKU"),
                      on="UNIQUE_FAMILY_CODE", how="left")
                .join(stock_parent, on=["PARENT_DEALER_CODE", "SKU"], how="left")
                .filter(~F.col("NUM_ACTIVE_SKUS").is_null())
                # Drop families for which we couldn't find any active SKU count —
                # these are unclassified/retired families that shouldn't get an SOQ.
            )


            # =================================================================
            # BLOCK 4d — ECR AGGREGATION FOR SOQ PROPORTION CALCULATION
            #
            # Pull the last 3 months of ECR data (from today's run date) and
            # compute:
            #   1. DEALER_FAMILY_CODE_NET_SALES — total sales per dealer+family
            #      (used to check if the family had any activity)
            #   2. DEALER_SKU_SALES — total sales per dealer+family+SKU
            #      (used to compute each SKU's proportion within the family)
            # =================================================================

            run_dt    = datetime.datetime.strptime(RUN_DATE, '%Y%m%d').replace(day=1)
            ecr_start = (run_dt - relativedelta(months=3)).strftime("%Y-%m-%d")
            ecr_end   = run_dt.strftime("%Y-%m-%d")

            # Pull ECR and enrich exactly as in Block 3b (OBD + family + parent dealer)
            ecr_soq = (
                session.table(ECR_TABLE)
                .filter(F.col("X_CUSTOMER_TYPE").isin(list(CUSTOMER_TYPE)))
                .filter(F.col("CAL_DATE") >= F.lit(ecr_start))
                .filter(F.col("CAL_DATE") <  F.lit(ecr_end))
                .with_column("NET_SALES",
                             F.col("INVOICED_SALES") + F.col("CANCELLED_SALES") + F.col("RETURNED_SALES"))
                .join(active_skus.select("SKU", "MODEL", "UNIQUE_FAMILY_CODE"),
                      on=["SKU", "MODEL"], how="left")
                .join(parent_dealer_mapping, on="DEALER_CODE", how="left")
            )

            if IS_OBD:
                obd = (
                    session.table(OBD_MAPPING_TABLE)
                    .select(F.col("PREVIOUS_OBD_SKU").alias("SKU"), F.col("CURRENT_OBD_SKU"))
                )
                ecr_soq = (
                    ecr_soq
                    .join(obd, on="SKU", how="left")
                    .with_column("SKU", F.coalesce(F.col("CURRENT_OBD_SKU"), F.col("SKU")))
                    .drop("CURRENT_OBD_SKU")
                )

            # Family-level total (to see if the dealer sold ANYTHING for this family)
            dealer_family_sales = (
                ecr_soq
                .group_by(["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE"])
                .agg(F.sum("NET_SALES").alias("DEALER_FAMILY_CODE_NET_SALES"))
            )

            # SKU-level totals (to compute each SKU's share of the family)
            dealer_family_sku_sales = (
                ecr_soq
                .group_by(["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU"])
                .agg(F.sum("NET_SALES").alias("DEALER_SKU_SALES"))
            )

            # Attach both aggregates to the SOQ data
            soq_data_sku = (
                soq_data_sku
                .join(dealer_family_sales,
                      on=["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE"], how="left")
                .with_column("DEALER_FAMILY_CODE_NET_SALES",
                             F.coalesce(F.col("DEALER_FAMILY_CODE_NET_SALES"), F.lit(0.0)))
                .join(dealer_family_sku_sales,
                      on=["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU"], how="left")
            )


            # =================================================================
            # BLOCK 4e — COMPUTE ACTIVE SKU FAMILY TOTAL AND PROPORTION
            #
            # Same three-case proportion logic as Block 3f, applied here
            # specifically to the SOQ calculation (using 3-month ECR, not the
            # cross-joined lookback window used in the TFT disaggregation).
            # =================================================================

            # Sum active SKU sales within each family using a window function.
            # This total = denominator for proportion.
            soq_window = Window.partition_by("PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE")

            soq_data_sku = soq_data_sku.with_column(
                "TOTAL_DEALER_ACTIVE_SKU_SALES",
                F.sum(F.coalesce(F.col("DEALER_SKU_SALES"), F.lit(0.0))).over(soq_window)
            ).with_column(
                "DEALER_SKU_SALES",
                F.coalesce(F.col("DEALER_SKU_SALES"), F.lit(0.0))
            )

            soq_data_sku = soq_data_sku.with_column(
                "PERCENT_PROPORTION",
                F.round(
                    F.when(
                        F.col("TOTAL_DEALER_ACTIVE_SKU_SALES") == F.lit(0),
                        F.lit(1.0) / F.col("NUM_ACTIVE_SKUS")      # no history: equal split
                    ).when(
                        F.col("DEALER_SKU_SALES") == F.lit(0),
                        F.lit(0.0)                                  # SKU had no sales: gets 0
                    ).otherwise(
                        F.col("DEALER_SKU_SALES") / F.col("TOTAL_DEALER_ACTIVE_SKU_SALES")
                    ),
                    5
                )
            )

            # Apply proportion to get SKU-level predicted sales for SOQ
            soq_data_sku = soq_data_sku.with_column(
                "PREDICTED_SALES_SKU",
                F.when(
                    F.col("NUM_ACTIVE_SKUS") == F.lit(1),
                    F.col("PREDICTED_SALES_TFT")          # single SKU: no split needed
                ).otherwise(
                    F.col("PERCENT_PROPORTION") * F.col("PREDICTED_SALES_TFT")
                )
            )


            # =================================================================
            # BLOCK 4f — JOIN TRANSIT LEAD TIMES
            #
            # Attach MAX_LEAD_TIME, MIN_LEAD_TIME, AVG_LEAD_TIME per
            # PARENT_DEALER_CODE + SKU from the pre-computed transit view.
            # These tell us how long delivery takes, which feeds into the
            # final SOQ formula (coverage = stock + in-transit + SOQ >= demand).
            # =================================================================

            transit = session.table(TRANSIT_TABLE)

            soq_final = soq_data_sku.join(
                transit, on=["PARENT_DEALER_CODE", "SKU"], how="left"
            )


            # =================================================================
            # BLOCK 4g — ATTACH METADATA AND SAVE
            #
            # Drop the intermediate denominator columns that were only needed
            # during calculation. Add run metadata for traceability.
            # =================================================================

            # Remove intermediate denominator columns (not needed in final output)
            cols_to_drop = [c for c in soq_final.columns
                            if c in ["DEALER_ACTIVE_SKU_TOTAL_SALES",
                                     "TOTAL_DEALER_ACTIVE_SKU_SALES"]]
            if cols_to_drop:
                soq_final = soq_final.drop(*cols_to_drop)

            soq_final = (
                soq_final
                .drop_duplicates()
                .with_column("STOCK_DATE_PERIOD", F.lit(stock_period))
                .with_column("STOCK_DATE",        F.lit(stock_date))
                .with_column("PLANNING_MONTH",    F.lit(planning_month))
                .with_column("RUN_DATE",          F.lit(RUN_DATE))
                .with_column("RUN_VERSION",       F.lit(RUN_VERSION))
                .with_column("IS_OBD",            F.lit(obd_flag))
            )

            soq_final.write.mode("append").save_as_table(BASE_SOQ_TABLE)
            print(f"SOQ base table updated: {BASE_SOQ_TABLE} | Stock period: {stock_period}")


    # =========================================================================
    # BLOCK 5 — DEMAND VARIABILITY
    #
    # Measures how erratic each dealer-SKU (and dealer-family) sales series is.
    # Metric = standard deviation of month-over-month changes in NET_SALES
    #          over the trailing 12 months.
    #
    # Why first differences (deltas) instead of raw std dev?
    # A series that grows steadily has high raw std dev but low delta std dev.
    # Delta std dev captures genuine VOLATILITY, not trend — which is what
    # matters for safety stock and SOQ buffer decisions.
    #
    # If a series has fewer than 2 deltas (i.e. only 1 or 2 months of data),
    # stddev is NULL. We default to 1 in that case.
    # =========================================================================

    for planning_month in MONTHS:
        print(f"\n===== Demand Variability for {planning_month} =====")

        run_dt     = datetime.datetime.strptime(RUN_DATE, "%Y%m%d").replace(day=1)
        dv_start   = (run_dt - relativedelta(months=12)).strftime("%Y-%m-%d")
        dv_end     = run_dt.strftime("%Y-%m-%d")

        # Pull 12 months of ECR, enrich with family code + parent dealer + OBD
        ecr_dv = (
            session.table(ECR_TABLE)
            .filter(F.col("X_CUSTOMER_TYPE").isin(list(CUSTOMER_TYPE)))
            .filter(F.col("CAL_DATE") >= F.lit(dv_start))
            .filter(F.col("CAL_DATE") <  F.lit(dv_end))
            .with_column("NET_SALES",
                         F.col("INVOICED_SALES") + F.col("CANCELLED_SALES") + F.col("RETURNED_SALES"))
            .join(active_skus.select("SKU", "MODEL", "UNIQUE_FAMILY_CODE"),
                  on=["SKU", "MODEL"], how="left")
            .join(parent_dealer_mapping, on="DEALER_CODE", how="left")
        )

        if IS_OBD:
            obd = session.table(OBD_MAPPING_TABLE).select(
                F.col("PREVIOUS_OBD_SKU").alias("SKU"), F.col("CURRENT_OBD_SKU")
            )
            ecr_dv = (
                ecr_dv
                .join(obd, on="SKU", how="left")
                .with_column("SKU", F.coalesce(F.col("CURRENT_OBD_SKU"), F.col("SKU")))
                .drop("CURRENT_OBD_SKU")
            )

        # --- SKU-level demand variability ---
        monthly_sku = (
            ecr_dv
            .with_column("CAL_MONTH", F.date_trunc("MONTH", F.to_date(F.col("CAL_DATE"))))
            .group_by(["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU", "CAL_MONTH"])
            .agg(F.sum("NET_SALES").alias("NET_SALES"))
        )

        # Lag window: previous month's sales for the same series
        sku_window = Window.partition_by(
            "PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU"
        ).order_by("CAL_MONTH")

        monthly_sku = monthly_sku \
            .with_column("PREV_NET_SALES", F.lag("NET_SALES").over(sku_window)) \
            .with_column("DELTA", F.col("NET_SALES") - F.col("PREV_NET_SALES")) \
            .filter(F.col("PREV_NET_SALES").is_not_null())  # first row has no lag; drop it

        dv_sku = (
            monthly_sku
            .group_by(["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "SKU"])
            .agg(F.stddev("DELTA").alias("DEMAND_VARIABILITY"))
            .with_column("DEMAND_VARIABILITY",
                         F.coalesce(F.col("DEMAND_VARIABILITY"), F.lit(1.0)))
            .with_column("PLANNING_MONTH", F.lit(planning_month))
            .with_column("ECR_START_DATE", F.lit(dv_start))
            .with_column("ECR_END_DATE",   F.lit(dv_end))
            .with_column("RUN_DATE",       F.lit(RUN_DATE))
            .with_column("RUN_VERSION",    F.lit(RUN_VERSION))
            .with_column("IS_OBD",         F.lit(obd_flag))
        )

        dv_sku.write.mode("append").save_as_table(DEMAND_VARIABILITY_TABLE)
        print(f"SKU-level demand variability saved to: {DEMAND_VARIABILITY_TABLE}")

        # --- Family-level demand variability (same logic, no SKU grouping) ---
        monthly_family = (
            ecr_dv
            .with_column("CAL_MONTH", F.date_trunc("MONTH", F.to_date(F.col("CAL_DATE"))))
            .group_by(["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE", "CAL_MONTH"])
            .agg(F.sum("NET_SALES").alias("NET_SALES"))
        )

        family_window_dv = Window.partition_by(
            "PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE"
        ).order_by("CAL_MONTH")

        monthly_family = monthly_family \
            .with_column("PREV_NET_SALES", F.lag("NET_SALES").over(family_window_dv)) \
            .with_column("DELTA", F.col("NET_SALES") - F.col("PREV_NET_SALES")) \
            .filter(F.col("PREV_NET_SALES").is_not_null())

        dv_family = (
            monthly_family
            .group_by(["PARENT_DEALER_CODE", "UNIQUE_FAMILY_CODE"])
            .agg(F.stddev("DELTA").alias("DEMAND_VARIABILITY"))
            .with_column("DEMAND_VARIABILITY",
                         F.coalesce(F.col("DEMAND_VARIABILITY"), F.lit(1.0)))
            .with_column("PLANNING_MONTH", F.lit(planning_month))
            .with_column("ECR_START_DATE", F.lit(dv_start))
            .with_column("ECR_END_DATE",   F.lit(dv_end))
            .with_column("RUN_DATE",       F.lit(RUN_DATE))
            .with_column("RUN_VERSION",    F.lit(RUN_VERSION))
            .with_column("IS_OBD",         F.lit(obd_flag))
        )

        dv_family.write.mode("append").save_as_table(DEMAND_VARIABILITY_FAMILY_TABLE)
        print(f"Family-level demand variability saved to: {DEMAND_VARIABILITY_FAMILY_TABLE}")

    # Return the family demand variability table as the worksheet result
    return session.table(DEMAND_VARIABILITY_FAMILY_TABLE)
