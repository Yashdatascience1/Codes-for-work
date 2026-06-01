import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col, lit, concat, split_part
from snowflake.snowpark import functions as F

# =============================================================================
# CONFIGURATION
# All key dates and table paths are defined here so nothing is hardcoded below.
# =============================================================================

START_DATE  = '2023-04-01'   # Earliest date of ECR sales data to pull
MAX_DATE    = '2027-04-01'   # Latest date in the date spine (covers future forecast horizon)
TRAIN_END_DATE = '2026-04-01'  # Everything before this is training data

CUSTOMER_TYPE_TO_CONSIDER = ['Individual']  # Filter: only retail individual customers

# SAP transit table and the mapping from shipping-point code (VSTEL) to plant code
TRANSIT_TABLE_NAME = "SAP_DATA_DATABASE.SAP_DATA.ZSDTTPTACTFRT"
VSTEL_MAPPING = {
    "HHDS": "HHHD",
    "HHGS": "HHHG",
    "HHUS": "HHHU",
    "HS4N": "HM4N",
    "HS5V": "HM5V",
    "HS6C": "HM6C"
}

# Snowflake table paths where intermediate and final outputs are saved
TRAIN_DATA_TABLE          = "MOP_DATABASE.SOQ.TRAIN_DATA_MONTHLY_DEALER_MODEL_FAMILY_CODE_MAR_2026_UPDATED_V18"
SKU_SUPERCEDENCE_MODEL_FAMILY = 'MOP_DATABASE.SOQ.SKU_SUPERCEDENCE_MODEL_FAMILY_MAR_2026_UPDATED_V18'
FESTIVE_TABLES            = 'MOP_DATABASE.SOQ.FESTIVE_DAYS_SOQ'
FESTIVE_PROPORTION_TABLE  = 'WORK_DATABASE.MOP.FESTIVE_INDIAN_SEASON_AGG_MONTH'
FINAL_TABLE               = "MOP_DATABASE.SOQ.TRAIN_AND_TEST_DATA_FOR_TFT"


# =============================================================================
# ENTRY POINT
# Snowflake Worksheets call main(session) automatically.
# Everything below runs sequentially inside this one function.
# =============================================================================

def main(session: snowpark.Session):

    # =========================================================================
    # BLOCK 1 — BUILD THE SKU SUPERCEDENCE TABLE
    #
    # Purpose: Create a unified lookup that maps every SKU to its Model Family
    # and a structured "Model Family Code" (e.g. ACTIVA<>DRUM<>SELF<>ALLOY<>RED).
    # This table is used downstream to enrich ECR sales rows with product hierarchy.
    #
    # Source tables:
    #   SKU_SUPERCEDENCE     — maps SKU -> MODEL + UNIQUEFAMILYCODE
    #   MODEL_FAMILY_MAPPING — maps MODEL -> MODEL_FAMILY (a higher grouping)
    # =========================================================================

    # Load the raw SKU-to-family mapping
    sku_data    = session.table("MOP_DATABASE.SOQ.SKU_SUPERCEDENCE")
    family_data = session.table("MOP_DATABASE.SOQ.MODEL_FAMILY_MAPPING")

    # Left join on MODEL so every SKU row gets its MODEL_FAMILY label.
    # SKUs whose MODEL doesn't exist in the mapping keep all their own columns
    # but get NULLs for MODEL_FAMILY — those are typically new/unclassified SKUs.
    sku_supercedence = sku_data.join(family_data, on="MODEL", how="left")

    # Preserve the original UNIQUEFAMILYCODE in a separate column before we
    # overwrite it below. This lets us trace back to the raw value if needed.
    sku_supercedence = sku_supercedence.with_column(
        "SKU_UNIQUE_FAMILY_CODE", col("UNIQUEFAMILYCODE")
    )

    # Snowpark sometimes wraps column names in double-quotes when doing joins.
    # Strip them so all downstream col() references work cleanly.
    for old_col in sku_supercedence.columns:
        new_col = old_col.replace('"', '')
        sku_supercedence = sku_supercedence.rename(old_col, new_col)

    # Build MODEL_FAMILY_CODE by combining the MODEL_FAMILY prefix with the
    # attribute string that follows the first '<>' in UNIQUEFAMILYCODE.
    #
    # Example:
    #   MODEL_FAMILY   = "ACTIVA"
    #   UNIQUEFAMILYCODE = "HONDA<>DRUM<>SELF<>ALLOY<>RED"
    #                                 ^--- everything after the first '<>'
    #   MODEL_FAMILY_CODE = "ACTIVA<>DRUM<>SELF<>ALLOY<>RED"
    #
    # Why? UNIQUEFAMILYCODE has the raw model name before the first '<>', which
    # might differ from the standardised MODEL_FAMILY label. We replace that
    # prefix so the code is consistent with the family hierarchy.
    sku_supercedence = sku_supercedence.with_column(
        "MODEL_FAMILY_CODE",
        F.concat(
            F.col("MODEL_FAMILY"),
            F.lit('<>'),
            # charindex finds the position of '<>' in the string.
            # Adding 2 skips past the two-character delimiter itself.
            F.substring(
                F.col("UNIQUEFAMILYCODE"),
                F.charindex(F.lit('<>'), F.col("UNIQUEFAMILYCODE")) + lit(2)
            )
        )
    )

    # Rename UNIQUEFAMILYCODE to the human-readable "UNIQUE FAMILY CODE"
    # (with a space) to match the convention used in other tables.
    sku_supercedence = sku_supercedence.rename("UNIQUEFAMILYCODE", "UNIQUE FAMILY CODE")

    # Persist this enriched mapping to Snowflake so it can be reused by
    # the ECR aggregation step (and other pipelines) without recomputing.
    sku_supercedence.write.mode("overwrite").save_as_table(SKU_SUPERCEDENCE_MODEL_FAMILY)


    # =========================================================================
    # BLOCK 2 — TRANSIT DATA PIPELINE
    #
    # Purpose: Compute how many days it takes for goods to travel from each
    # plant to each dealer. This produces three outputs:
    #   a) TRANSIT_DEALER_MAPPING_NEW           — transit time per dealer (all plants)
    #   b) TRANSIT_DEALER_SKU_PLANT_MAPPING_NEW — transit time per dealer + SKU + plant
    #   c) PARENT_DEALER_TRANSIT_TIME_SKU_NEW   — aggregated view (max/min/avg per SKU)
    # =========================================================================

    # --- 2a. Load and clean the raw SAP transit table ---

    # The SAP table stores one row per customer (dealer) per shipping point
    # with the transit days (LDAYS) and the validity date (DATB).
    transit_raw = session.table(TRANSIT_TABLE_NAME)

    # Keep only the four columns we actually need
    transit_raw = transit_raw.select("KUNNR", "VSTEL", "DATB", "LDAYS")

    # DATB = "00000000" is SAP's sentinel for "no valid date". Drop those rows
    # because they represent incomplete or dummy records.
    transit_raw = transit_raw.filter(col("DATB") != lit("00000000"))

    # SAP stores customer numbers (KUNNR) with leading zeros (e.g. "0001234").
    # Our dealer codes don't have leading zeros, so strip them for the join later.
    transit_raw = transit_raw.with_column("Customer", F.ltrim(col("KUNNR"), lit("0")))

    # VSTEL is a shipping-point code from SAP. We need the plant code (PLANT)
    # that corresponds to it. Build a CASE-style expression from VSTEL_MAPPING.
    # For any VSTEL not in the mapping, PLANT will be an empty string "".
    plant_expr = F.lit("")
    for vstel_val, plant_val in VSTEL_MAPPING.items():
        plant_expr = F.iff(col("VSTEL") == lit(vstel_val), lit(plant_val), plant_expr)
    transit_raw = transit_raw.with_column("PLANT", plant_expr)

    # Drop rows where KUNNR is null (no customer = no useful record)
    transit_raw = transit_raw.filter(col("KUNNR").is_not_null())

    # Each customer/VSTEL/PLANT combination can have multiple historical rows
    # (one per validity date). We only want the most recent one (highest DATB).
    # Step 1: compute the max DATB per group
    max_datb = transit_raw.group_by("Customer", "VSTEL", "PLANT") \
                          .agg(F.max("DATB").alias("MAX_DATE"))

    # Step 2: join back and keep only the rows where DATB equals that maximum
    transit_raw = transit_raw.join(max_datb, on=["Customer", "VSTEL", "PLANT"], how="left") \
                             .filter(col("DATB") == col("MAX_DATE"))

    # Rename columns to business-friendly names and cast LDAYS to integer
    transit_data = transit_raw.select(
        col("Customer").alias("DEALER_CODE"),
        col("LDAYS").cast("INTEGER").alias("TRANSIT_TIME"),
        col("PLANT"),
        col("VSTEL")
    )

    # --- 2b. Map dealer codes to their parent dealer ---
    # Dealers roll up to a PARENT_DEALER_CODE.
    # PAR_ORG_NAME looks like "DELHI-NORTH" — the parent code is the part before "-".

    parent_dealer_mapping = session.table("FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH") \
        .filter(col("X_DEALER_CODE_HIER").is_not_null()) \
        .select(
            col("X_DEALER_CODE_HIER").alias("DEALER_CODE"),
            col("PAR_ORG_NAME")
        ).distinct() \
        .with_column("PARENT_DEALER_CODE", split_part(col("PAR_ORG_NAME"), lit("-"), lit(1)))

    # Join transit data with parent dealer mapping on DEALER_CODE.
    # Inner join: if a dealer has no parent mapping, drop it — we can't attribute
    # the transit time to any parent, making it useless for forecasting.
    transit_with_parent = transit_data.join(parent_dealer_mapping, on="DEALER_CODE", how="inner")

    # Save: transit time per dealer (irrespective of SKU)
    transit_with_parent.write.mode("overwrite").save_as_table(
        "MOP_DATABASE.SOQ.TRANSIT_DEALER_MAPPING_NEW"
    )

    # --- 2c. Enrich with SKU-to-plant mapping ---
    # Each plant produces certain SKUs. This table tells us which SKUs come
    # from which plant, so we can link transit time to a specific SKU.

    sku_plant_mapping = session.table("MOP_DATABASE.SOQ.SKU_PLANT_MAPPING_VIEW")

    # Inner join on PLANT: only keep combinations where we know which SKUs
    # the plant produces. Records with unmapped plants are dropped.
    transit_with_sku = transit_with_parent.join(sku_plant_mapping, on="PLANT", how="inner")

    # Save: transit time per dealer + SKU + plant (most granular transit table)
    transit_with_sku.write.mode("overwrite").save_as_table(
        "MOP_DATABASE.SOQ.TRANSIT_DEALER_SKU_PLANT_MAPPING_NEW"
    )

    # --- 2d. Aggregate transit times per SKU + Parent Dealer ---
    # The model needs a single lead-time number per SKU-dealer combination.
    # We compute max, min, and average so the model can use whichever suits it.
    # Filter to PRODUCTION=TRUE to exclude testing/dummy plant-SKU combinations.

    agg_transit = session.table("MOP_DATABASE.SOQ.TRANSIT_DEALER_SKU_PLANT_MAPPING_NEW") \
        .filter(col("PRODUCTION") == lit(True)) \
        .group_by("SKU", "PARENT_DEALER_CODE") \
        .agg(
            F.max("TRANSIT_TIME").alias("MAX_LEAD_TIME"),
            F.min("TRANSIT_TIME").alias("MIN_LEAD_TIME"),
            F.avg("TRANSIT_TIME").alias("AVG_LEAD_TIME")
        )

    # Save as a view (not a table) — views are cheap to refresh and always
    # reflect the latest data in TRANSIT_DEALER_SKU_PLANT_MAPPING_NEW
    agg_transit.create_or_replace_view(
        "MOP_DATABASE.SOQ.PARENT_DEALER_TRANSIT_TIME_SKU_NEW"
    )


    # =========================================================================
    # BLOCK 3 — PULL AND CLEAN ECR SALES DATA
    #
    # Purpose: Get actual retail sales (ECR = Electronic Channel Retails) from
    # the transactional table, enrich with OBD mapping and SKU supercedence,
    # and aggregate to monthly level per dealer + model family.
    # =========================================================================

    # --- 3a. Pull raw ECR sales ---
    # CUSTOMER_RETAILS has one row per SKU per dealer per day.
    # Filter to Individual customers only (no institutional/bulk orders).
    # Filter from START_DATE onwards (we don't need older history).

    ecr_sales = session.table("ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS") \
        .filter(col("X_CUSTOMER_TYPE").in_(CUSTOMER_TYPE_TO_CONSIDER)) \
        .filter(col("CAL_DATE") >= lit(START_DATE))

    # NET_SALES = Invoiced - Cancelled - Returned (note: CANCELLED and RETURNED
    # are stored as negative numbers in the source, so addition gives the net)
    ecr_sales = ecr_sales.with_column(
        "NET_SALES",
        col("INVOICED_SALES") + col("CANCELLED_SALES") + col("RETURNED_SALES")
    )

    # --- 3b. Apply OBD (Order-Based Distribution) SKU mapping ---
    # Some SKUs get superseded by a new SKU under OBD. If a sale was recorded
    # under an old SKU, we remap it to the current OBD SKU before aggregating.
    # This ensures the model trains on consistent, forward-looking SKU codes.

    obd_data  = session.table("MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW")

    # Left join: keep all ECR rows; attach the CURRENT_OBD_SKU where available
    ecr_sales = ecr_sales.join(obd_data, ecr_sales["SKU"] == obd_data["PREVIOUS_OBD_SKU"], how="left")

    # If CURRENT_OBD_SKU is populated, use it; otherwise keep the original SKU.
    # This is the core remapping step.
    ecr_sales = ecr_sales.with_column(
        "SKU", F.coalesce(col("CURRENT_OBD_SKU"), col("SKU"))
    )

    # --- 3c. Join with SKU supercedence to get the product family hierarchy ---
    # We need MODEL_FAMILY and MODEL_FAMILY_CODE for aggregation.
    # Inner join: ECR rows that can't be matched to a known SKU/MODEL combination
    # are dropped. This is intentional — unclassified SKUs can't be forecast.

    sku_map   = session.table(SKU_SUPERCEDENCE_MODEL_FAMILY)
    ecr_sales = ecr_sales.join(sku_map, ["MODEL", "SKU"], how="inner")

    # --- 3d. Map each dealer to its parent dealer ---
    # Forecasting happens at the PARENT_DEALER level (a regional grouping),
    # not the individual dealer level (too granular, too sparse).

    parent_map = session.table("FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH") \
        .select(col("X_DEALER_CODE_HIER").alias("DEALER_CODE"), col("PAR_ORG_NAME"))

    ecr_sales = ecr_sales.join(parent_map, "DEALER_CODE", how="left")

    # PAR_ORG_NAME looks like "DELHI-NORTH"; split on "-" to get just "DELHI"
    ecr_sales = ecr_sales.with_column(
        "PARENT_DEALER_CODE", split_part(col("PAR_ORG_NAME"), lit("-"), lit(1))
    )

    # Remove exact duplicate rows that could arise from the multiple joins above
    ecr_sales = ecr_sales.distinct()

    # --- 3e. Aggregate to monthly grain ---
    # The TFT model trains on one row per PARENT_DEALER + MODEL_FAMILY_CODE + Month.
    # Sum NET_SALES across all dealers and SKUs within each group.

    monthly_agg = ecr_sales.group_by(
        "PARENT_DEALER_CODE",
        "MODEL_FAMILY",
        "MODEL_FAMILY_CODE",
        F.year("CAL_DATE").alias("CAL_YEAR"),
        F.month("CAL_DATE").alias("CAL_MONTH")
    ).agg(F.sum("NET_SALES").alias("NET_SALES"))

    # Reconstruct a proper DATE column from the year and month integers.
    # All dates are set to the 1st of the month (standard for monthly time series).
    monthly_agg = monthly_agg.with_column(
        "Date", F.date_from_parts(col("CAL_YEAR"), col("CAL_MONTH"), lit(1))
    )


    # =========================================================================
    # BLOCK 4 — CREATE A COMPLETE DATE SPINE AND FILL SALES GAPS
    #
    # Purpose: The aggregated sales data has holes — months where a dealer sold
    # zero units of a model family simply don't appear. The TFT model requires
    # a continuous, unbroken time series. We must explicitly add those zero rows.
    #
    # Approach:
    #   1. Get every unique PARENT_DEALER + MODEL_FAMILY_CODE combination
    #   2. Generate one row per month from START_DATE to MAX_DATE
    #   3. Cross-join (cartesian) the two → a complete grid of all possible rows
    #   4. Left-join actual sales onto the grid → missing months get NULL → fill with 0
    # =========================================================================

    # Step 1: All unique series (dealer + product family combinations)
    unique_combos = monthly_agg.select(
        "PARENT_DEALER_CODE", "MODEL_FAMILY", "MODEL_FAMILY_CODE"
    ).distinct()

    # Step 2: Build the date spine — one DATE row per month from START to MAX
    total_months_query = f"SELECT DATEDIFF(month, '{START_DATE}', '{MAX_DATE}') as months"
    total_months = session.sql(total_months_query).collect()[0]['MONTHS']

    # session.range() generates sequential integers 0, 1, 2, ..., total_months.
    # add_months() shifts START_DATE forward by that many months.
    date_spine = session.range(total_months + 1).select(
        F.add_months(F.to_date(lit(START_DATE)), col("ID")).alias("DATE")
    )

    # Step 3: Cross-join to get every combination of series × month.
    # No join key = full cartesian product. This is intentional and correct here.
    master_grid = unique_combos.join(date_spine)

    # Step 4: Left-join actual sales onto the grid.
    # Rows in the grid with no matching sale will have NULL in NET_SALES.
    final_data = master_grid.join(
        monthly_agg,
        on=["PARENT_DEALER_CODE", "MODEL_FAMILY", "MODEL_FAMILY_CODE", "DATE"],
        how="left"
    )

    # Create a composite series identifier used by the TFT model as a unique key
    # per time series. Format: "DELHI<>ACTIVA<>DRUM<>SELF<>ALLOY<>RED"
    final_data = final_data.with_column(
        "PARENT_DEALER_CODE_MODEL_FAMILY",
        concat(F.trim(col("PARENT_DEALER_CODE")), lit("<>"), F.trim(col("MODEL_FAMILY_CODE")))
    )

    # Drop redundant columns — CAL_YEAR and CAL_MONTH came from the aggregation
    # step but we now have DATE, so these are no longer needed
    final_data = final_data.drop("CAL_YEAR", "CAL_MONTH")

    # Unpack MODEL_FAMILY_CODE into individual product attribute columns.
    # MODEL_FAMILY_CODE = "ACTIVA<>DRUM<>SELF<>ALLOY<>RED"
    #   Part 1 → MODEL_NAME    (e.g. ACTIVA)
    #   Part 2 → BRAKE_TYPE    (e.g. DRUM)
    #   Part 3 → IGNITION_TYPE (e.g. SELF)
    #   Part 4 → WHEEL_TYPE    (e.g. ALLOY)
    #   Part 5 → COLOUR        (e.g. RED)
    # These are used as static covariates in the TFT model.
    final_data = final_data \
        .with_column("MODEL_NAME",    split_part(col("MODEL_FAMILY_CODE"), lit("<>"), lit(1))) \
        .with_column("BRAKE_TYPE",    split_part(col("MODEL_FAMILY_CODE"), lit("<>"), lit(2))) \
        .with_column("IGNITION_TYPE", split_part(col("MODEL_FAMILY_CODE"), lit("<>"), lit(3))) \
        .with_column("WHEEL_TYPE",    split_part(col("MODEL_FAMILY_CODE"), lit("<>"), lit(4))) \
        .with_column("COLOUR",        split_part(col("MODEL_FAMILY_CODE"), lit("<>"), lit(5)))

    # Replace NULL NET_SALES (gap months) with 0. Cast ensures type consistency.
    final_data = final_data.with_column(
        "NET_SALES", F.coalesce(col("NET_SALES"), lit(0).cast("DECIMAL(38,6)"))
    )

    # Rename DATE to MONTH_OF_SALE for clarity in downstream steps
    final_data = final_data.rename("DATE", "MONTH_OF_SALE")


    # =========================================================================
    # BLOCK 5 — JOIN FESTIVE / SEASONAL FEATURES
    #
    # Purpose: Add festive calendar features to every row. These are known
    # external factors that influence two-wheeler demand in India (Diwali,
    # Navratri, wedding season, etc.) and serve as time-varying covariates.
    #
    # Two festive tables:
    #   FESTIVE_TABLES           — day counts per festival per month
    #   FESTIVE_PROPORTION_TABLE — proportion of the month that each festive
    #                              phase covers (used as a continuous feature)
    # =========================================================================

    # Load the festive day counts table and standardise column names.
    # The source uses "DATE" but we join on "MONTH_OF_SALE", so rename it.
    festive_df = session.table(FESTIVE_TABLES)
    for old_col in festive_df.columns:
        clean_col = old_col.replace('"', '').upper()
        festive_df = festive_df.rename(old_col, 'MONTH_OF_SALE' if clean_col == "DATE" else clean_col)

    # Load the festive proportion table. MONTH_DATE is the same concept as
    # MONTH_OF_SALE so rename it for a clean join.
    festive_proportion_df = session.table(FESTIVE_PROPORTION_TABLE) \
                                   .rename("MONTH_DATE", "MONTH_OF_SALE")

    # Left join festive features onto the sales data.
    # Months with no matching festive record will get NULLs — handled next.
    final_set = final_data.join(festive_df, on="MONTH_OF_SALE", how="left")
    final_set = final_set.join(festive_proportion_df, on="MONTH_OF_SALE", how="left")


    # =========================================================================
    # BLOCK 6 — VALIDATE AND STANDARDISE THE FINAL DATASET
    #
    # Purpose: Before saving, make sure the data is clean and model-ready.
    #   1. Standardise all column names (UPPER_CASE_WITH_UNDERSCORES)
    #   2. Grain check: each series+month must have exactly 1 NET_SALES value
    #   3. Replace NULLs in festive columns with 0
    # =========================================================================

    # Step 1: Standardise column names
    # Removes quotes, uppercases everything, replaces spaces and hyphens with underscores
    for c in final_set.columns:
        new_name = c.replace('"', '').upper().replace(' ', '_').replace('-', '_')
        if c != new_name:
            final_set = final_set.with_column_renamed(c, new_name)

    # Step 2: Grain check — each (series, month) pair must appear exactly once.
    # Duplicates would distort training targets and must be caught before saving.
    grain_check = final_set.group_by("PARENT_DEALER_CODE_MODEL_FAMILY", "MONTH_OF_SALE") \
                           .agg(F.count("*").alias("ROW_COUNT")) \
                           .filter(col("ROW_COUNT") > 1)

    duplicate_count = grain_check.count()
    if duplicate_count > 0:
        # Show the offending rows to help debug the source of duplicates
        grain_check.show(5)
        raise ValueError(
            f"Data Integrity Error: {duplicate_count} duplicate series-month pairs detected. "
            "Check the OBD join or SKU supercedence for conflicting mappings."
        )

    # Step 3: Replace NULLs in festive columns with 0.
    # NULLs appear when a month has no record in the festive tables (e.g. future months).
    # For the model, 0 = no festive activity, which is the correct interpretation.
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
        if c in final_set.columns:
            final_set = final_set.with_column(c, F.coalesce(col(c), lit(0)))


    # =========================================================================
    # BLOCK 7 — SAVE THE FINAL DATASET
    #
    # This is the training + test dataset for the TFT model.
    # It contains one row per (parent dealer, model family code, month) with:
    #   - NET_SALES as the target variable
    #   - Product attributes as static covariates (MODEL_NAME, BRAKE_TYPE, etc.)
    #   - Festive features as time-varying known covariates
    # =========================================================================

    final_set.write.mode("overwrite").save_as_table(FINAL_TABLE)

    return f"Pipeline complete. Final table saved to {FINAL_TABLE}."
