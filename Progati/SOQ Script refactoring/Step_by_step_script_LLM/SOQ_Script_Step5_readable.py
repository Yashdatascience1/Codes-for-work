import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import pandas as pd
import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_DATE    = datetime.datetime.today().strftime('%Y%m%d')
RUN_VERSION = 33

# Source: the full SOQ output table produced by Step 4
SOQ_SOURCE_TABLE     = 'MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_2'

# Reference: dealer master containing AREA_OFFICE and ZONE geography attributes
DEALER_MAPPING_TABLE = 'ANALYTICS_DATABASE.ANALYTICS_SALES.VW_DEALER_MASTER'

# Destination: SOQ data enriched with geographic breakdown for reporting
OUTPUT_TABLE = 'MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_WITH_AO_VERSION_4'


def main(session: snowpark.Session):

    # =========================================================================
    # BLOCK 1 — LOAD THE SOQ OUTPUT FROM STEP 4
    #
    # Pull all rows for today's run. The filter on RUN_DATE + RUN_VERSION ensures
    # we only pick up the current execution, not any historical runs stored in
    # the same table.
    # =========================================================================

    soq_query = f"""
        SELECT * FROM {SOQ_SOURCE_TABLE}
        WHERE RUN_DATE = '{RUN_DATE}'
          AND RUN_VERSION = {RUN_VERSION}
    """
    soq_data = session.sql(soq_query).to_pandas()
    print(f"SOQ rows loaded: {len(soq_data)}")


    # =========================================================================
    # BLOCK 2 — PRESERVE AND DROP IS_OBD BEFORE THE MERGE
    #
    # IS_OBD is a metadata flag ('Y'/'N') that must survive the join.
    # We extract it as a Python list before the merge and re-attach it after.
    #
    # Why? A left join on PARENT_DEALER_CODE can produce a perfectly 1:1 match,
    # but if pandas shifts row order or produces any NaN during the merge,
    # directly re-assigning from a list is safer than trusting the column
    # to come through the join cleanly (especially with mixed dtype columns).
    # =========================================================================

    is_obd = soq_data['IS_OBD'].tolist()   # save as list before dropping
    soq_data.drop(columns=['IS_OBD'], inplace=True)


    # =========================================================================
    # BLOCK 3 — LOAD DEALER GEOGRAPHIC ATTRIBUTES
    #
    # VW_DEALER_MASTER contains AREA_OFFICE (e.g. "Mumbai") and ZONE
    # (e.g. "West") for each dealer. These are used in downstream reports
    # to break SOQ figures by geography.
    #
    # The source column is DEALER_CODE, but the SOQ table uses PARENT_DEALER_CODE
    # (already rolled up to parent level in Step 3). Rename to match.
    # =========================================================================

    dealer_data = (
        session.table(DEALER_MAPPING_TABLE)
        .select(col("DEALER_CODE"), col("AREA_OFFICE"), col("ZONE"))
        .to_pandas()
    )
    # Rename DEALER_CODE -> PARENT_DEALER_CODE to match the join key in soq_data
    dealer_data.columns = ['PARENT_DEALER_CODE', 'AREA_OFFICE', 'ZONE']
    print(f"Dealer geo mapping rows: {len(dealer_data)}")


    # =========================================================================
    # BLOCK 4 — JOIN GEOGRAPHIC ATTRIBUTES ONTO SOQ DATA
    #
    # Left join: every SOQ row is kept.
    # Dealers not present in the master get NULLs in AREA_OFFICE and ZONE —
    # this is acceptable; they will show up as blanks in the report rather than
    # being silently dropped.
    # =========================================================================

    soq_data = pd.merge(soq_data, dealer_data, on='PARENT_DEALER_CODE', how='left')
    print(f"Rows after geo join: {len(soq_data)}")


    # =========================================================================
    # BLOCK 5 — RE-ATTACH IS_OBD AND SAVE
    #
    # Put IS_OBD back as the last column (matches the original table structure).
    # Append to the output table — don't overwrite, so multiple runs accumulate.
    # =========================================================================

    soq_data['IS_OBD'] = is_obd

    session.create_dataframe(soq_data).write.mode("append").save_as_table(OUTPUT_TABLE)
    print(f"Written {len(soq_data)} rows to {OUTPUT_TABLE}")

    return session.create_dataframe(soq_data)
