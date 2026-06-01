import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime
import logging
import io

# =============================================================================
# CONFIGURATION
# =============================================================================

PLANNING_MONTH  = "2026-05-01"
USE_OBD_MAPPING = False
CUSTOMER_TYPES  = ["Individual"]

# Table paths
ECR_SALES_TABLE       = "ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS"
OBD_MAPPING_TABLE     = "MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW"
DEALER_MAPPING_TABLE  = "ANALYTICS_DATABASE.ANALYTICS_SALES.VW_DEALER_MASTER"
PARENT_DEALER_TABLE   = "FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH"
OUTPUT_TABLE          = "MOP_DATABASE.SOQ.RETAIL_ALL_MONTHS"


# =============================================================================
# LOGGER SETUP
# =============================================================================

def _setup_logger():
    log_stream = io.StringIO()
    logger = logging.getLogger("SOQ_Step6")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger, log_stream


# =============================================================================
# BLOCK A — OBD MAPPING
# =============================================================================

def fetch_obd_mapping(session, logger):
    """
    Loads the OBD (Order-Based Distribution) SKU mapping table.
    Adds a SKU column aliased from PREVIOUS_OBD_SKU so the DataFrame can be
    joined on 'SKU' directly.
    """
    obd_df = session.table(OBD_MAPPING_TABLE).to_pandas()
    obd_df['SKU'] = obd_df['PREVIOUS_OBD_SKU']
    logger.info("OBD mapping loaded | Rows: %s", len(obd_df))
    return obd_df


# =============================================================================
# BLOCK B — ECR SALES DATA
# =============================================================================

def get_filtered_ecr_sales(session, logger):
    """
    Pulls 3 months of ECR sales data ending the day before PLANNING_MONTH.
    Aggregates to DEALER_CODE + MODEL + SKU + CAL_DATE grain in SQL to reduce
    data transfer volume before pulling to pandas.

    If USE_OBD_MAPPING is True, remaps old SKU variants to current OBD SKU
    so that sales are credited to the live product code.

    Returns a DataFrame with a NET_SALES column computed as:
        NET_SALES = INVOICED_SALES + CANCELLED_SALES + RETURNED_SALES
    (CANCELLED and RETURNED are stored as negative values in the source.)
    """
    planning_dt = datetime.datetime.strptime(PLANNING_MONTH, "%Y-%m-%d").date()
    start_date  = (planning_dt - relativedelta(months=3)).replace(day=1)
    end_date    = planning_dt - relativedelta(days=1)

    types_sql = ",".join(f"'{t}'" for t in CUSTOMER_TYPES)

    query = f"""
        SELECT DEALER_CODE, MODEL, SKU, CAL_DATE,
               SUM(INVOICED_SALES)   AS INVOICED_SALES,
               SUM(CANCELLED_SALES)  AS CANCELLED_SALES,
               SUM(RETURNED_SALES)   AS RETURNED_SALES
        FROM {ECR_SALES_TABLE}
        WHERE X_CUSTOMER_TYPE IN ({types_sql})
          AND CAL_DATE BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        GROUP BY DEALER_CODE, MODEL, SKU, CAL_DATE
    """
    ecr_df = session.sql(query).to_pandas()
    logger.info("ECR data loaded | Period: %s to %s | Rows: %s", start_date, end_date, len(ecr_df))

    if USE_OBD_MAPPING:
        obd_df = fetch_obd_mapping(session, logger)
        ecr_df = ecr_df.merge(obd_df, on='SKU', how='left')
        ecr_df['CURRENT_OBD_SKU'] = ecr_df['CURRENT_OBD_SKU'].fillna(ecr_df['SKU'])
        ecr_df = ecr_df.drop(columns=['SKU']).rename(columns={'CURRENT_OBD_SKU': 'SKU'})
        logger.info("OBD remapping applied to ECR data.")

    ecr_df['NET_SALES'] = (
        ecr_df['INVOICED_SALES'].fillna(0) +
        ecr_df['CANCELLED_SALES'].fillna(0) +
        ecr_df['RETURNED_SALES'].fillna(0)
    )

    return ecr_df


# =============================================================================
# BLOCK C — PARENT DEALER MAPPING
# =============================================================================

def get_parent_dealer_mapping(session, logger):
    """
    Returns a DataFrame mapping individual DEALER_CODE -> PARENT_DEALER_CODE.
    PAR_ORG_NAME looks like "DELHI-NORTH"; splitting on "-" gives "DELHI".
    """
    query = f"""
        SELECT DISTINCT X_DEALER_CODE_HIER AS DEALER_CODE, PAR_ORG_NAME
        FROM {PARENT_DEALER_TABLE}
        WHERE X_DEALER_CODE_HIER IS NOT NULL
    """
    mapping_df = session.sql(query).to_pandas()
    mapping_df['PARENT_DEALER_CODE'] = (
        mapping_df['PAR_ORG_NAME'].apply(lambda x: str(x).split("-")[0].strip())
    )
    logger.info("Parent dealer mapping loaded | Unique dealers: %s", len(mapping_df))
    return mapping_df[['DEALER_CODE', 'PARENT_DEALER_CODE']]


# =============================================================================
# BLOCK D — DEALER GEOGRAPHIC ATTRIBUTES
# =============================================================================

def get_dealer_geo_attributes(session, logger):
    """
    Pulls AREA_OFFICE and ZONE from the dealer master view.
    Renames DEALER_CODE to PARENT_DEALER_CODE to align with the ECR join key.
    """
    dealer_df = (
        session.table(DEALER_MAPPING_TABLE)
        .select(col("DEALER_CODE"), col("AREA_OFFICE"), col("ZONE"))
        .to_pandas()
    )
    dealer_df.columns = ['PARENT_DEALER_CODE', 'AREA_OFFICE', 'ZONE']
    logger.info("Dealer geo attributes loaded | Rows: %s", len(dealer_df))
    return dealer_df


# =============================================================================
# BLOCK E — ABC CLASSIFICATION
# =============================================================================

def generate_final_abc(ecr_df, planning_month, logger):
    """
    Classifies each (dealer, SKU) pair into A / B / C based on cumulative
    contribution to the dealer's total sales.

    Steps:
      1. Aggregate NET_SALES to DEALER_CODE + SKU grain
      2. Compute each dealer's total sales
      3. Calculate each SKU's fractional contribution (PERCENTILE)
      4. Sort high-to-low within each dealer, cumsum to get CUM_PERCENTILE
      5. Assign A (<=70%), B (<=90%), C (>90%)

    Note: thresholds are fractional (0-1) here, unlike Step 4 which used
    percentage scale (0-100). Both are correct — the boundary logic differs.
    """
    # Step 1: total sales per dealer + SKU
    agg = (
        ecr_df.groupby(["DEALER_CODE", "SKU"], as_index=False)
              .agg({"NET_SALES": "sum"})
              .rename(columns={"NET_SALES": "DEALER_SKU_SALES"})
    )

    # Step 2: dealer total
    total_sales = (
        agg.groupby("DEALER_CODE", as_index=False)
           .agg({"DEALER_SKU_SALES": "sum"})
           .rename(columns={"DEALER_SKU_SALES": "DEALER_TOTAL_SALES"})
    )

    # Step 3: fractional contribution
    merged = pd.merge(agg, total_sales, on="DEALER_CODE")
    merged["PERCENTILE"] = merged["DEALER_SKU_SALES"] / merged["DEALER_TOTAL_SALES"]

    # Step 4: sort and cumulative sum
    merged = merged.sort_values(by=["DEALER_CODE", "PERCENTILE"], ascending=[True, False])
    merged["CUM_PERCENTILE"] = merged.groupby("DEALER_CODE")["PERCENTILE"].cumsum()

    # Step 5: assign label
    def assign_abc(cum_pct):
        if cum_pct <= 0.70:
            return "A"
        elif cum_pct <= 0.90:
            return "B"
        else:
            return "C"

    merged["FINAL_ABC"]      = merged["CUM_PERCENTILE"].apply(assign_abc)
    merged["PLANNING_MONTH"] = planning_month

    result = merged[[
        "PLANNING_MONTH", "DEALER_CODE", "SKU",
        "DEALER_SKU_SALES", "DEALER_TOTAL_SALES",
        "PERCENTILE", "CUM_PERCENTILE", "FINAL_ABC"
    ]]

    abc_dist = result["FINAL_ABC"].value_counts().to_dict()
    logger.info("ABC classification complete | Rows: %s | Distribution: %s",
                len(result), abc_dist)

    return result


# =============================================================================
# MAIN
# =============================================================================

def main(session: snowpark.Session):
    logger, log_stream = _setup_logger()
    logger.info("========== SOQ Step 6 Pipeline Started ==========")

    try:
        # Load ECR sales for the 3-month lookback window
        ecr_data = get_filtered_ecr_sales(session, logger)

        # Attach PARENT_DEALER_CODE (inner join: drop dealers not in the hierarchy)
        parent_mapping = get_parent_dealer_mapping(session, logger)
        ecr_data = pd.merge(ecr_data, parent_mapping, on="DEALER_CODE", how="inner")
        logger.info("ECR rows after parent dealer inner join: %s", len(ecr_data))

        # Attach AREA_OFFICE and ZONE for geographic reporting
        dealer_geo = get_dealer_geo_attributes(session, logger)
        ecr_data = pd.merge(ecr_data, dealer_geo, on="PARENT_DEALER_CODE", how="left")
        logger.info("ECR rows after geo join: %s", len(ecr_data))

        # Compute ABC classification and save
        final_abc_df = generate_final_abc(ecr_data, PLANNING_MONTH, logger)
        session.create_dataframe(final_abc_df).write.mode("append").save_as_table(OUTPUT_TABLE)
        logger.info("Appended %s rows to %s", len(final_abc_df), OUTPUT_TABLE)

        logger.info("========== SOQ Step 6 Pipeline Complete ==========")

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        raise

    finally:
        print(log_stream.getvalue())

    return session.create_dataframe(final_abc_df)
