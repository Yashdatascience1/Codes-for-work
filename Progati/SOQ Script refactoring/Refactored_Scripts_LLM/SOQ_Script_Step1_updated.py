import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col, lit, concat, split_part, coalesce
from snowflake.snowpark import functions as F
import logging
import io

# --- CONFIGURATION ---
START_DATE = '2023-04-01' 
MAX_DATE = "2027-04-01" 
TRAIN_END_DATE = '2026-04-01' 
CUSTOMER_TYPE_TO_CONSIDER = ['Individual'] 

# TRANSIT TABLE CONFIG
TRANSIT_TABLE_NAME = "SAP_DATA_DATABASE.SAP_DATA.ZSDTTPTACTFRT"
VSTEL_MAPPING = {"HHDS": "HHHD", "HHGS": "HHHG", "HHUS": "HHHU", "HS4N": "HM4N", "HS5V": "HM5V", "HS6C": "HM6C"}

# TABLE PATHS
TRAIN_DATA_TABLE = "MOP_DATABASE.SOQ.TRAIN_DATA_MONTHLY_DEALER_MODEL_FAMILY_CODE_MAR_2026_UPDATED_V18" 
SKU_SUPERCEDENCE_MODEL_FAMILY = 'MOP_DATABASE.SOQ.SKU_SUPERCEDENCE_MODEL_FAMILY_MAR_2026_UPDATED_V18' 
FESTIVE_TABLES = 'MOP_DATABASE.SOQ.FESTIVE_DAYS_SOQ' 
FESTIVE_PROPORTION_TABLE = 'WORK_DATABASE.MOP.FESTIVE_INDIAN_SEASON_AGG_MONTH' 
FINAL_TABLE = "MOP_DATABASE.SOQ.TRAIN_AND_TEST_DATA_FOR_TFT" 
LOG_STAGE_PATH = "@MOP_DATABASE.SOQ.PIPELINE_LOGS/ecr_validation_audit.txt"

# --- REFACTORED SNOWPARK FUNCTIONS ---

def fetchSKUSupercedence_snowpark(session):
    data = session.table("MOP_DATABASE.SOQ.SKU_SUPERCEDENCE") 
    data_1 = session.table("MOP_DATABASE.SOQ.MODEL_FAMILY_MAPPING") 
    result = data.join(data_1, on="MODEL", how="left") 
    result = result.with_column("SKU_UNIQUE_FAMILY_CODE", col("UNIQUEFAMILYCODE")) 

    for old_col in result.columns:
        new_col = old_col.replace('"','')
        result = result.rename(old_col, new_col) 
    
    result = result.with_column("MODEL_FAMILY_CODE",
                            F.concat(F.col("MODEL_FAMILY"), F.lit('<>'),
                            F.substring(F.col("UNIQUEFAMILYCODE"),
                                F.charindex(F.lit('<>'), F.col("UNIQUEFAMILYCODE")) + lit(2))
                                )) 
    
    result = result.rename("UNIQUEFAMILYCODE", "UNIQUE FAMILY CODE") 
    result.write.mode("overwrite").save_as_table(SKU_SUPERCEDENCE_MODEL_FAMILY) 
    return result

def get_ecr_sales_snowpark(session, customer_types, start_date):
    ecr_sales = session.table("ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS") \
        .filter(col("X_CUSTOMER_TYPE").in_(customer_types)) \
        .filter(col("CAL_DATE") >= lit(start_date)) 
    
    ecr_sales = ecr_sales.with_column("NET_SALES", 
        col("INVOICED_SALES") + col("CANCELLED_SALES") + col("RETURNED_SALES")) 
    return ecr_sales

def process_ecr_aggregation_snowpark(session, agg_type, customer_types, start_date):
    ecr_sales = get_ecr_sales_snowpark(session, customer_types, start_date) 
    obd_data = session.table("MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW") 
    ecr_sales = ecr_sales.join(obd_data, ecr_sales["SKU"] == obd_data["PREVIOUS_OBD_SKU"], how="left") 
    ecr_sales = ecr_sales.with_column("SKU", F.coalesce(col("CURRENT_OBD_SKU"), col("SKU"))) 
    
    sku_map = session.table(SKU_SUPERCEDENCE_MODEL_FAMILY) 
    ecr_sales = ecr_sales.join(sku_map, ["MODEL", "SKU"], how="inner") 
    
    parent_map = session.table("FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH") \
        .select(col("X_DEALER_CODE_HIER").alias("DEALER_CODE"), col("PAR_ORG_NAME")) 
    ecr_sales = ecr_sales.join(parent_map, "DEALER_CODE", how="left") 
    ecr_sales = ecr_sales.with_column("PARENT_DEALER_CODE", split_part(col("PAR_ORG_NAME"), lit("-"), lit(1))) 

    ecr_sales = ecr_sales.distinct()
    
    if agg_type == "monthly":
        final_agg = ecr_sales.group_by(
            "PARENT_DEALER_CODE", "MODEL_FAMILY", "MODEL_FAMILY_CODE", 
            F.year("CAL_DATE").alias("CAL_YEAR"), 
            F.month("CAL_DATE").alias("CAL_MONTH")
        ).agg(F.sum("NET_SALES").alias("NET_SALES")) 
        final_agg = final_agg.with_column("Date", F.date_from_parts(col("CAL_YEAR"), col("CAL_MONTH"), lit(1))) 
    return final_agg

def create_date_spine(session, start_date_str, max_date_str, logger):
    logger.info("🛠️ Creating Date Spine: Range from %s to %s", start_date_str, max_date_str) 
    count_query = f"SELECT DATEDIFF(month, '{start_date_str}', '{max_date_str}') as months"
    total_months = session.sql(count_query).collect()[0]['MONTHS'] 
    
    date_spine = session.range(total_months + 1).select(
        F.add_months(F.to_date(lit(start_date_str)), col("ID")).alias("DATE")
    ) 
    logger.info("✅ Date Spine generated with %s monthly intervals.", total_months + 1) 
    return date_spine

def fill_sales_gaps_snowpark(session, final_agg, start_date, max_date, logger):
    unique_combos = final_agg.select("PARENT_DEALER_CODE", "MODEL_FAMILY", "MODEL_FAMILY_CODE").distinct() 
    date_spine = create_date_spine(session, start_date, max_date, logger) 
    
    master_grid = unique_combos.join(date_spine) 
    final_data = master_grid.join(final_agg, 
        on=["PARENT_DEALER_CODE", "MODEL_FAMILY", "MODEL_FAMILY_CODE", "DATE"], 
        how="left") 

    final_data = final_data.with_column("PARENT_DEALER_CODE_MODEL_FAMILY", 
        concat(F.trim(col("PARENT_DEALER_CODE")), lit("<>"), F.trim(col("MODEL_FAMILY_CODE")))) 

    final_data = final_data.drop("CAL_YEAR","CAL_MONTH")

    final_data = final_data.with_column("MODEL_NAME", split_part(col("MODEL_FAMILY_CODE"), lit("<>"), lit(1))) \
                           .with_column("BRAKE_TYPE", split_part(col("MODEL_FAMILY_CODE"), lit("<>"), lit(2))) \
                           .with_column("IGNITION_TYPE", split_part(col("MODEL_FAMILY_CODE"), lit("<>"), lit(3))) \
                           .with_column("WHEEL_TYPE", split_part(col("MODEL_FAMILY_CODE"), lit("<>"), lit(4))) \
                           .with_column("COLOUR", split_part(col("MODEL_FAMILY_CODE"), lit("<>"), lit(5))) 

    final_data = final_data.with_column("NET_SALES", F.coalesce(col("NET_SALES"), lit(0).cast("DECIMAL(38,6)"))) 
    return final_data.rename("DATE", "MONTH_OF_SALE") 

def perform_final_dataset_validation(df, logger):
    logger.info("🏁 Starting Final Dataset Validation & Standardization...") 
    
    # Standardize Column Names (Upper Case and Underscores)
    for c in df.columns:
        new_name = c.replace('"', '').upper().replace(' ', '_').replace('-', '_')
        if c != new_name:
            df = df.with_column_renamed(c, new_name) 
    
    # 1. Grain Check: Ensure 1 value of NET_SALES per Series/Month combination
    grain_check = df.group_by("PARENT_DEALER_CODE_MODEL_FAMILY", "MONTH_OF_SALE") \
                    .agg(F.count("*").alias("ROW_COUNT")).filter(col("ROW_COUNT") > 1) 
    
    duplicate_count = grain_check.count()
    if duplicate_count > 0:
        logger.error("❌ GRAIN VALIDATION FAILED: Found %s duplicate combinations!", duplicate_count) 
        grain_check.show(5)
        raise ValueError("Data Integrity Error: Duplicate pairs detected.") 
    else:
        logger.info("✅ GRAIN VALIDATION PASSED: Only 1 value of NET_SALES per Dealer-Model-Month.")

    # 2. Log Row and Feature (Column) counts
    total_rows = df.count()
    num_features = len(df.columns)
    logger.info("📊 Dataset Summary: Total Rows = %s | Total Features = %s", total_rows, num_features)

    # Null Replacement for Festive Columns
    festive_columns = [
        'AKSHAYA_TRITIYA_DAYS', 'BHAI_DOOJ_DAYS', 'BUDDHA_PURNIMA_DAYS', 'CHHATH_PUJA_DAYS',
        'DHANTERAS_DAYS', 'DIWALI_DAYS', 'DUSSEHRA_(VIJAYADASHAMI)_DAYS', 'EID_UL_FITR_DAYS',
        'GANESH_CHATURTHI_DAYS', 'GANGA_DUSSEHRA_DAYS', 'GOVARDHAN_POOJA_DAYS', 'GURU_PURNIMA_DAYS',
        'HANUMAN_JAYANTI_DAYS', 'HARTALIK_TEEJ_DAYS', 'HOLI_DAYS', 'HOLIKA_DAHAN_DAYS',
        'JAGANNATH_RATHYATRA_DAYS', 'JANMASHTAMI_DAYS', 'KARWA_CHAUTH_DAYS', 'LOHRI_DAYS',
        'MAHA_SHIVARATRI_DAYS', 'MAKAR_SANKRANTI_PONGAL_DAYS', 'NAG_PANCHAMI_DAYS', 'NAVRATRI_DAYS',
        'NEW_YEAR_DAYS', 'ONAM_DAYS', 'PITRAPAKSHA_DAYS', 'RAKSHA_BANDHAN_DAYS', 'REPUBLIC_DAY_DAYS',
        'VASANT_PANCHAMI_DAYS', 'VISHWAKARMA_PUJA_DAYS', 'MARRIAGE_DAYS', 'FESTIVE_PHASE_I',
        'FESTIVE_PHASE_II', 'FESTIVE_PHASE_III', 'PITRU_PAKSH', 'YEAR',
        'TOTAL_DAYS_FESTIVE_PHASE_I', 'TOTAL_DAYS_FESTIVE_PHASE_II', 'TOTAL_DAYS_FESTIVE_PHASE_III',
        'TOTAL_DAYS_PITRU_PAKSH', 'PROP_FESTIVE_PHASE_I', 'PROP_EVENT_FESTIVE_PHASE_I',
        'PROP_FESTIVE_PHASE_II', 'PROP_EVENT_FESTIVE_PHASE_II', 'PROP_FESTIVE_PHASE_III',
        'PROP_EVENT_FESTIVE_PHASE_III', 'PROP_PITRU_PAKSH', 'PROP_EVENT_PITRU_PAKSH'
    ] 
    
    for c in festive_columns:
        if c in df.columns:
            df = df.with_column(c, F.coalesce(col(c), lit(0))) 
            
    logger.info("✅ Final validation and Null replacement complete.") 
    return df


# --- TRANSIT TABLE FUNCTIONS ---

def fetchTransitData_snowpark(session):
    """
    Fetches and processes transit data from the SAP transit table.
    Filters out invalid dates, applies VSTEL-to-PLANT mapping, keeps only
    the most recent record per Customer/VSTEL/PLANT, and returns a clean
    Snowpark DataFrame with columns: DEALER_CODE, TRANSIT_TIME, PLANT, VSTEL.
    """
    transit_table = session.table(TRANSIT_TABLE_NAME)

    # Keep only required columns and filter out invalid date sentinels
    transit_data = transit_table.select("KUNNR", "VSTEL", "DATB", "LDAYS") \
                                .filter(col("DATB") != lit("00000000"))

    # Strip leading zeros from customer code to align with dealer code format
    transit_data = transit_data.with_column("Customer", F.ltrim(col("KUNNR"), lit("0")))

    # Apply VSTEL -> PLANT mapping using a CASE expression built from VSTEL_MAPPING
    case_expr = F.lit("")
    for vstel_val, plant_val in VSTEL_MAPPING.items():
        case_expr = F.iff(col("VSTEL") == lit(vstel_val), lit(plant_val), case_expr)
    transit_data = transit_data.with_column("PLANT", case_expr)

    # Drop rows where KUNNR is null
    transit_data = transit_data.filter(col("KUNNR").is_not_null())

    # Keep only the most recent record per Customer/VSTEL/PLANT (max DATB)
    max_date = transit_data.group_by("Customer", "VSTEL", "PLANT") \
                           .agg(F.max("DATB").alias("MAX_DATE"))
    transit_data = transit_data.join(max_date, on=["Customer", "VSTEL", "PLANT"], how="left") \
                               .filter(col("DATB") == col("MAX_DATE"))

    # Rename and cast to final schema
    transit_data = transit_data.select(
        col("Customer").alias("DEALER_CODE"),
        col("LDAYS").cast("INTEGER").alias("TRANSIT_TIME"),
        col("PLANT"),
        col("VSTEL")
    )

    return transit_data


def fetchParentDealerMapping_snowpark(session):
    """
    Returns a Snowpark DataFrame mapping DEALER_CODE -> PARENT_DEALER_CODE.
    """
    return session.table("FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH") \
        .filter(col("X_DEALER_CODE_HIER").is_not_null()) \
        .select(
            col("X_DEALER_CODE_HIER").alias("DEALER_CODE"),
            col("PAR_ORG_NAME")
        ).distinct() \
        .with_column("PARENT_DEALER_CODE", split_part(col("PAR_ORG_NAME"), lit("-"), lit(1)))


def fetchSkuPlantMapping_snowpark(session):
    """
    Returns a Snowpark DataFrame with the SKU-to-PLANT mapping.
    """
    return session.table("MOP_DATABASE.SOQ.SKU_PLANT_MAPPING_VIEW")


def fetchAggregateTransitTime_snowpark(session):
    """
    Returns a Snowpark DataFrame with MAX, MIN, and AVG transit time
    aggregated per SKU and PARENT_DEALER_CODE, filtered to PRODUCTION=TRUE rows.
    """
    return session.table("MOP_DATABASE.SOQ.TRANSIT_DEALER_SKU_PLANT_MAPPING_NEW") \
        .filter(col("PRODUCTION") == lit(True)) \
        .group_by("SKU", "PARENT_DEALER_CODE") \
        .agg(
            F.max("TRANSIT_TIME").alias("MAX_LEAD_TIME"),
            F.min("TRANSIT_TIME").alias("MIN_LEAD_TIME"),
            F.avg("TRANSIT_TIME").alias("AVG_LEAD_TIME")
        )


def run_transit_pipeline(session, logger):
    """
    Orchestrates the full transit data pipeline:
      1. Fetch and clean raw transit records
      2. Join with parent dealer mapping  -> save TRANSIT_DEALER_MAPPING_NEW
      3. Join with SKU-plant mapping      -> save TRANSIT_DEALER_SKU_PLANT_MAPPING_NEW
      4. Aggregate transit times          -> create view PARENT_DEALER_TRANSIT_TIME_SKU_NEW
    """
    logger.info("Transit Data Pipeline starting...")

    # Step 1: Raw transit data
    transit_data = fetchTransitData_snowpark(session)

    # Step 2: Join with parent dealer mapping and persist
    parent_dealer_mapping = fetchParentDealerMapping_snowpark(session)
    transit_with_parent = transit_data.join(parent_dealer_mapping, on="DEALER_CODE", how="inner")
    transit_with_parent.write.mode("overwrite").save_as_table("MOP_DATABASE.SOQ.TRANSIT_DEALER_MAPPING_NEW")
    logger.info("Saved TRANSIT_DEALER_MAPPING_NEW")

    # Step 3: Join with SKU-plant mapping and persist
    sku_plant_mapping = fetchSkuPlantMapping_snowpark(session)
    transit_with_sku = transit_with_parent.join(sku_plant_mapping, on="PLANT", how="inner")
    transit_with_sku.write.mode("overwrite").save_as_table("MOP_DATABASE.SOQ.TRANSIT_DEALER_SKU_PLANT_MAPPING_NEW")
    logger.info("Saved TRANSIT_DEALER_SKU_PLANT_MAPPING_NEW")

    # Step 4: Aggregate and create view
    agg_transit = fetchAggregateTransitTime_snowpark(session)
    agg_transit.create_or_replace_view("MOP_DATABASE.SOQ.PARENT_DEALER_TRANSIT_TIME_SKU_NEW")
    logger.info("Created view PARENT_DEALER_TRANSIT_TIME_SKU_NEW")

    logger.info("Transit Data Pipeline complete.")

# --- MAIN WORKSHEET FUNCTION ---

def main(session: snowpark.Session):
    # 1. Setup In-Memory Logging
    log_stream = io.StringIO()
    logger = logging.getLogger("ECR_Pipeline")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    try:
        # 2. Execute Data Pipeline
        sku_supercedence = fetchSKUSupercedence_snowpark(session)

        # Transit Table Pipeline
        run_transit_pipeline(session, logger)

        final_agg = process_ecr_aggregation_snowpark(session, 'monthly', CUSTOMER_TYPE_TO_CONSIDER, START_DATE)
        
        # Fill Gaps and Feature Extraction
        final_data = fill_sales_gaps_snowpark(session, final_agg, START_DATE, MAX_DATE, logger)
        
        # Festive Data Join
        festive_df = session.table(FESTIVE_TABLES)
        for old_col in festive_df.columns:
            new_col = old_col.replace('"','').upper()
            festive_df = festive_df.rename(old_col, 'MONTH_OF_SALE' if new_col == "DATE" else new_col) 
            
        festive_proportion_df = session.table(FESTIVE_PROPORTION_TABLE).rename("MONTH_DATE", "MONTH_OF_SALE") 
        
        # Merge all
        final_set = final_data.join(festive_df, on="MONTH_OF_SALE", how="left") 
        final_set = final_set.join(festive_proportion_df, on="MONTH_OF_SALE", how="left") 
        
        # 3. Final Validation (Includes Row, Feature, and Grain checks)
        final_training_set = perform_final_dataset_validation(final_set, logger)
        
        # 4. Save to Snowflake
        final_training_set.write.mode("overwrite").save_as_table(FINAL_TABLE) 
        logger.info("🚀 Pipeline Successful. Table %s created.", FINAL_TABLE)
        
    except Exception as e:
        logger.error("❌ Pipeline Failed: %s", str(e))
        raise e
    finally:
        # 5. Upload Log to Stage
        log_content = log_stream.getvalue()
        session.file.put_stream(
            io.BytesIO(log_content.encode()), 
            LOG_STAGE_PATH, 
            overwrite=True
        )
    
    return "Pipeline completed. Log uploaded to Stage."