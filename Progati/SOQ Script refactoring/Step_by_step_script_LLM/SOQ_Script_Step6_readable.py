import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

PLANNING_MONTH  = "2026-05-01"
USE_OBD_MAPPING = False           # set True to remap old SKU variants to current OBD SKU
CUSTOMER_TYPES  = ["Individual"]  # filter: retail customers only

# Table paths
ECR_SALES_TABLE      = "ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS"
OBD_MAPPING_TABLE    = "MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW"
DEALER_MAPPING_TABLE = "ANALYTICS_DATABASE.ANALYTICS_SALES.VW_DEALER_MASTER"
PARENT_DEALER_TABLE  = "FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH"
OUTPUT_TABLE         = "MOP_DATABASE.SOQ.RETAIL_ALL_MONTHS"


def main(session: snowpark.Session):

    # =========================================================================
    # BLOCK 1 — DETERMINE THE ECR DATE WINDOW
    #
    # We want the 3 months of ECR sales immediately before PLANNING_MONTH.
    # Example: PLANNING_MONTH = 2026-05-01
    #   start_date = 2026-02-01  (3 months back, snapped to 1st of month)
    #   end_date   = 2026-04-30  (last day before planning month starts)
    # =========================================================================

    planning_dt = datetime.datetime.strptime(PLANNING_MONTH, "%Y-%m-%d").date()
    start_date  = (planning_dt - relativedelta(months=3)).replace(day=1)
    end_date    = planning_dt - relativedelta(days=1)

    print(f"ECR window: {start_date} to {end_date}")


    # =========================================================================
    # BLOCK 2 — PULL ECR SALES DATA FROM SNOWFLAKE
    #
    # Pre-aggregate in SQL (GROUP BY) to reduce the volume pulled into pandas.
    # This avoids pulling millions of daily transaction rows when we only
    # need the DEALER + MODEL + SKU + DATE grain.
    #
    # NET_SALES = INVOICED + CANCELLED + RETURNED.
    # CANCELLED and RETURNED are stored as NEGATIVE values in the source, so
    # addition (not subtraction) gives the correct net.
    # =========================================================================

    types_sql = ",".join(f"'{t}'" for t in CUSTOMER_TYPES)

    ecr_query = f"""
        SELECT DEALER_CODE, MODEL, SKU, CAL_DATE,
               SUM(INVOICED_SALES)   AS INVOICED_SALES,
               SUM(CANCELLED_SALES)  AS CANCELLED_SALES,
               SUM(RETURNED_SALES)   AS RETURNED_SALES
        FROM {ECR_SALES_TABLE}
        WHERE X_CUSTOMER_TYPE IN ({types_sql})
          AND CAL_DATE BETWEEN DATE '{start_date}' AND DATE '{end_date}'
        GROUP BY DEALER_CODE, MODEL, SKU, CAL_DATE
    """
    ecr_data = session.sql(ecr_query).to_pandas()
    print(f"ECR rows loaded: {len(ecr_data)}")


    # =========================================================================
    # BLOCK 3 — OPTIONAL: REMAP OLD SKUs TO CURRENT OBD VARIANT
    #
    # If USE_OBD_MAPPING is True, replace each old SKU code with the current
    # OBD (Order-Based Distribution) SKU. This ensures sales of discontinued
    # variants are attributed to the live product code that replaced them.
    #
    # fillna(ecr['SKU']) = if no OBD mapping exists, keep the original SKU.
    # =========================================================================

    if USE_OBD_MAPPING:
        obd_df = session.table(OBD_MAPPING_TABLE).to_pandas()
        obd_df['SKU'] = obd_df['PREVIOUS_OBD_SKU']   # create join key

        ecr_data = ecr_data.merge(obd_df, on='SKU', how='left')
        ecr_data['CURRENT_OBD_SKU'] = ecr_data['CURRENT_OBD_SKU'].fillna(ecr_data['SKU'])
        ecr_data = ecr_data.drop(columns=['SKU']).rename(columns={'CURRENT_OBD_SKU': 'SKU'})
        print("OBD remapping applied.")

    # Compute NET_SALES (fillna handles any null individual components)
    ecr_data['NET_SALES'] = (
        ecr_data['INVOICED_SALES'].fillna(0) +
        ecr_data['CANCELLED_SALES'].fillna(0) +
        ecr_data['RETURNED_SALES'].fillna(0)
    )


    # =========================================================================
    # BLOCK 4 — ATTACH PARENT DEALER CODE
    #
    # Each dealer belongs to a PARENT_DEALER (regional group).
    # PAR_ORG_NAME looks like "DELHI-NORTH" — split on "-" gives "DELHI".
    #
    # Inner join: dealers not in the hierarchy table are dropped.
    # This is intentional — unregistered dealers have no parent grouping
    # and can't be attributed to any region.
    # =========================================================================

    parent_query = f"""
        SELECT DISTINCT X_DEALER_CODE_HIER AS DEALER_CODE, PAR_ORG_NAME
        FROM {PARENT_DEALER_TABLE}
        WHERE X_DEALER_CODE_HIER IS NOT NULL
    """
    parent_mapping = session.sql(parent_query).to_pandas()
    parent_mapping['PARENT_DEALER_CODE'] = (
        parent_mapping['PAR_ORG_NAME'].apply(lambda x: str(x).split("-")[0].strip())
    )
    parent_mapping = parent_mapping[['DEALER_CODE', 'PARENT_DEALER_CODE']]

    # Inner join: only keep ECR rows for dealers we can map to a parent
    ecr_data = pd.merge(ecr_data, parent_mapping, on="DEALER_CODE", how="inner")
    print(f"ECR rows after parent dealer join: {len(ecr_data)}")


    # =========================================================================
    # BLOCK 5 — ATTACH AREA OFFICE AND ZONE
    #
    # VW_DEALER_MASTER has AREA_OFFICE and ZONE per dealer.
    # These are retained for geographic reporting downstream (not used in ABC).
    # The source DEALER_CODE maps to PARENT_DEALER_CODE here because the
    # dealer master stores the parent-level identifier in DEALER_CODE.
    #
    # Left join: ECR rows for dealers not in the master keep NULLs in
    # AREA_OFFICE and ZONE — they appear as blanks in reports, not dropped.
    # =========================================================================

    dealer_geo = (
        session.table(DEALER_MAPPING_TABLE)
        .select(col("DEALER_CODE"), col("AREA_OFFICE"), col("ZONE"))
        .to_pandas()
    )
    dealer_geo.columns = ['PARENT_DEALER_CODE', 'AREA_OFFICE', 'ZONE']

    ecr_data = pd.merge(ecr_data, dealer_geo, on="PARENT_DEALER_CODE", how="left")
    print(f"ECR rows after geo join: {len(ecr_data)}")


    # =========================================================================
    # BLOCK 6 — ABC CLASSIFICATION AT DEALER × SKU LEVEL
    #
    # Goal: label each (dealer, SKU) pair as A, B, or C based on how much of
    # that dealer's total sales the SKU contributes.
    #
    # Method (Pareto / cumulative contribution):
    #   1. Sum NET_SALES per DEALER_CODE + SKU  -> DEALER_SKU_SALES
    #   2. Sum DEALER_SKU_SALES per DEALER_CODE -> DEALER_TOTAL_SALES (denominator)
    #   3. PERCENTILE = DEALER_SKU_SALES / DEALER_TOTAL_SALES
    #      (fractional contribution of this SKU to the dealer's total)
    #   4. Sort SKUs high-to-low within each dealer
    #   5. Cumulative sum of PERCENTILE (CUM_PERCENTILE)
    #      Running total tells us "the top N SKUs account for X% of sales"
    #   6. Assign ABC:
    #      A = CUM_PERCENTILE <= 0.70  (top 70% of revenue — fast movers)
    #      B = CUM_PERCENTILE <= 0.90  (next 20%)
    #      C = CUM_PERCENTILE >  0.90  (bottom 10% — slow movers / tail)
    #
    # Note: Step 4 uses the same logic but on FAMILY-level data with percentage
    # scale (0-100). This step is at SKU level with fractional scale (0-1).
    # =========================================================================

    # Step 1: aggregate NET_SALES to DEALER + SKU grain
    agg = (
        ecr_data.groupby(["DEALER_CODE", "SKU"], as_index=False)
                .agg({"NET_SALES": "sum"})
                .rename(columns={"NET_SALES": "DEALER_SKU_SALES"})
    )

    # Step 2: dealer-level total (denominator for proportion)
    dealer_totals = (
        agg.groupby("DEALER_CODE", as_index=False)
           .agg({"DEALER_SKU_SALES": "sum"})
           .rename(columns={"DEALER_SKU_SALES": "DEALER_TOTAL_SALES"})
    )

    # Step 3: merge and compute fractional contribution
    abc_data = pd.merge(agg, dealer_totals, on="DEALER_CODE")
    abc_data["PERCENTILE"] = abc_data["DEALER_SKU_SALES"] / abc_data["DEALER_TOTAL_SALES"]

    # Step 4: sort high-to-low within each dealer, then cumulative sum
    abc_data = abc_data.sort_values(
        by=["DEALER_CODE", "PERCENTILE"], ascending=[True, False]
    )
    abc_data["CUM_PERCENTILE"] = abc_data.groupby("DEALER_CODE")["PERCENTILE"].cumsum()

    # Step 5: assign ABC label based on cumulative threshold
    def assign_abc(cum_pct):
        if cum_pct <= 0.70:
            return "A"
        elif cum_pct <= 0.90:
            return "B"
        else:
            return "C"

    abc_data["FINAL_ABC"]      = abc_data["CUM_PERCENTILE"].apply(assign_abc)
    abc_data["PLANNING_MONTH"] = PLANNING_MONTH

    # Select only the columns needed for the output table
    final_abc_df = abc_data[[
        "PLANNING_MONTH", "DEALER_CODE", "SKU",
        "DEALER_SKU_SALES", "DEALER_TOTAL_SALES",
        "PERCENTILE", "CUM_PERCENTILE", "FINAL_ABC"
    ]]

    print(f"ABC rows computed: {len(final_abc_df)}")
    print(f"ABC distribution:\n{final_abc_df['FINAL_ABC'].value_counts()}")


    # =========================================================================
    # BLOCK 7 — SAVE
    #
    # Append to the output table. Using "append" so each planning month's
    # results accumulate — the table becomes a historical log of ABC
    # classifications across months.
    # =========================================================================

    session.create_dataframe(final_abc_df).write.mode("append").save_as_table(OUTPUT_TABLE)
    print(f"Written {len(final_abc_df)} rows to {OUTPUT_TABLE}")

    return session.create_dataframe(final_abc_df)
