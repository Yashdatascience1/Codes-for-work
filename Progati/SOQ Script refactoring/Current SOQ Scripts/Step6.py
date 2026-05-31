import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime
import logging

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
config = {
    "planning_month": "2026-05-01",
    "use_obd_mapping": False,
    "customer_types": ["Individual"],
    "database": {
        "ecr_sales_table": "ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS",
        "OBD_MAPPING": "MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW",
        "dealer_mapping": "ANALYTICS_DATABASE.ANALYTICS_SALES.VW_DEALER_MASTER",
        "parent_dealer_mapping": "FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH",
        "output_table": "MOP_DATABASE.SOQ.RETAIL_ALL_MONTHS"
    }
}

# OBD Mapping
def fetch_obd_mapping(session, table):
    obd_df = session.table(table).to_pandas()
    obd_df['SKU'] = obd_df['PREVIOUS_OBD_SKU']
    return obd_df

# ECR Sales Data
def get_filtered_ecr_sales(session, config):
    planning_month = datetime.datetime.strptime(config["planning_month"], "%Y-%m-%d").date()
    start_date = (planning_month - relativedelta(months=3)).replace(day=1)
    end_date = (planning_month - relativedelta(days=1))

    ecr_table = config["database"]["ecr_sales_table"]
    types = ",".join(f"'{t}'" for t in config["customer_types"])

    query = f"""
        SELECT DEALER_CODE, MODEL, SKU, CAL_DATE,
               SUM(INVOICED_SALES) AS INVOICED_SALES,
               SUM(CANCELLED_SALES) AS CANCELLED_SALES,
               SUM(RETURNED_SALES) AS RETURNED_SALES
        FROM {ecr_table}
        WHERE X_CUSTOMER_TYPE IN ({types})
          AND CAL_DATE BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        GROUP BY DEALER_CODE, MODEL, SKU, CAL_DATE
    """
    ecr_df = session.sql(query).to_pandas()

    if config.get("use_obd_mapping", False):
        obd_df = fetch_obd_mapping(session, config["database"]["OBD_MAPPING"])
        ecr_df = ecr_df.merge(obd_df, on="SKU", how="left")
        ecr_df['CURRENT_OBD_SKU'] = ecr_df['CURRENT_OBD_SKU'].fillna(ecr_df['SKU'])
        ecr_df = ecr_df.drop(columns=['SKU']).rename(columns={'CURRENT_OBD_SKU': 'SKU'})

    ecr_df['NET_SALES'] = (
        ecr_df['INVOICED_SALES'].fillna(0) +
        ecr_df['CANCELLED_SALES'].fillna(0) +
        ecr_df['RETURNED_SALES'].fillna(0)
    )

    return ecr_df

# Parent Dealer Mapping
def get_parent_dealer_mapping(session, table):
    query = f"""
        SELECT DISTINCT X_DEALER_CODE_HIER AS DEALER_CODE, PAR_ORG_NAME
        FROM {table}
        WHERE X_DEALER_CODE_HIER IS NOT NULL
    """
    mapping_df = session.sql(query).to_pandas()
    mapping_df['PARENT_DEALER_CODE'] = mapping_df['PAR_ORG_NAME'].apply(lambda x: str(x).split("-")[0].strip())
    logger.info(f'Retrieved {len(mapping_df)} parent dealer mappings')
    return mapping_df[['DEALER_CODE', 'PARENT_DEALER_CODE']]

# ABC Classification
def generate_final_abc(ecr_df: pd.DataFrame, planning_month: str) -> pd.DataFrame:
    # Step 1: Aggregate sales
    agg = (
        ecr_df.groupby(["DEALER_CODE", "SKU"], as_index=False)
              .agg({"NET_SALES": "sum"})
              .rename(columns={"NET_SALES": "DEALER_SKU_SALES"})
    )

    # Step 2: Dealer total sales
    total_sales = (
        agg.groupby("DEALER_CODE", as_index=False)
           .agg({"DEALER_SKU_SALES": "sum"})
           .rename(columns={"DEALER_SKU_SALES": "DEALER_TOTAL_SALES"})
    )

    # Step 3: Join & compute contribution
    merged = pd.merge(agg, total_sales, on="DEALER_CODE")
    merged["PERCENTILE"] = merged["DEALER_SKU_SALES"] / merged["DEALER_TOTAL_SALES"]

    # Step 4: Sort and compute cumulative contribution
    merged = merged.sort_values(by=["DEALER_CODE", "PERCENTILE"], ascending=[True, False])
    merged["CUM_PERCENTILE"] = merged.groupby("DEALER_CODE")["PERCENTILE"].cumsum()

    # Step 5: Assign ABC
    def assign_abc(cum_pct):
        if cum_pct <= 0.70:
            return "A"
        elif cum_pct <= 0.90:
            return "B"
        else:
            return "C"

    merged["FINAL_ABC"] = merged["CUM_PERCENTILE"].apply(assign_abc)
    merged["PLANNING_MONTH"] = planning_month

    return merged[["PLANNING_MONTH", "DEALER_CODE", "SKU", "DEALER_SKU_SALES", "DEALER_TOTAL_SALES" ,"PERCENTILE", "CUM_PERCENTILE", "FINAL_ABC"]]

# Main Function
def main(session: snowpark.Session):
    # Step 1: Get ECR Sales
    ecr_data = get_filtered_ecr_sales(session, config)

    # Step 2: Add PARENT_DEALER_CODE via mapping
    parent_dealer_mapping = get_parent_dealer_mapping(session, config["database"]["parent_dealer_mapping"])
    ecr_data = pd.merge(ecr_data, parent_dealer_mapping, on="DEALER_CODE", how="inner")

    
    # Step 3: Add AREA_OFFICE & ZONE (optional, not used in ABC but can be retained for extended views)
    dealer_df = session.table(config["database"]["dealer_mapping"]).select(
        col("DEALER_CODE"), col("AREA_OFFICE"), col("ZONE")
    ).to_pandas()
    dealer_df.columns = ['PARENT_DEALER_CODE', 'AREA_OFFICE', 'ZONE']
    ecr_data = pd.merge(ecr_data, dealer_df, on="PARENT_DEALER_CODE", how="left")

    # Step 4: Compute FINAL_ABC and save
    final_abc_df = generate_final_abc(ecr_data, config["planning_month"])
    final_df_sp = session.create_dataframe(final_abc_df)
    final_df_sp.write.mode("append").save_as_table(config["database"]["output_table"])

    logger.info(f"Appended {len(final_abc_df)} rows to {config['database']['output_table']}")
    return final_df_sp

   # 23854
