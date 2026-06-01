import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import pandas as pd
import datetime
import logging
import io

# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_DATE    = datetime.datetime.today().strftime('%Y%m%d')
RUN_VERSION = 33

# Source: the SOQ output table from Step 4, filtered to this run
SOQ_SOURCE_TABLE  = 'MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_2'

# Reference: dealer master view with AREA_OFFICE and ZONE attributes
DEALER_MAPPING_TABLE = 'ANALYTICS_DATABASE.ANALYTICS_SALES.VW_DEALER_MASTER'

# Destination: SOQ data enriched with geographic attributes
OUTPUT_TABLE = 'MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_WITH_AO_VERSION_4'


# =============================================================================
# LOGGER SETUP
# =============================================================================

def _setup_logger():
    log_stream = io.StringIO()
    logger = logging.getLogger("SOQ_Step5")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger, log_stream


# =============================================================================
# BLOCK A — LOAD SOQ DATA
# =============================================================================

def load_soq_data(session, run_date, run_version, logger):
    """
    Pulls this run's SOQ output from Step 4.
    IS_OBD is extracted before the merge because the dealer mapping join can
    introduce duplicate rows that corrupt list-based column reconstruction —
    storing it separately and re-attaching after the merge is the safe pattern.
    """
    query = f"""
        SELECT * FROM {SOQ_SOURCE_TABLE}
        WHERE RUN_DATE = '{run_date}'
          AND RUN_VERSION = {run_version}
    """
    soq_data = session.sql(query).to_pandas()
    logger.info("SOQ data loaded | Rows: %s | RUN_DATE: %s | RUN_VERSION: %s",
                len(soq_data), run_date, run_version)

    # Preserve IS_OBD as a list before dropping.
    # It gets re-attached after the join to avoid Pandas merge creating duplicate
    # or misaligned rows (the dealer mapping join is left-join and row-order safe).
    is_obd = soq_data['IS_OBD'].tolist()
    soq_data.drop(columns=['IS_OBD'], inplace=True)

    return soq_data, is_obd


# =============================================================================
# BLOCK B — LOAD DEALER GEOGRAPHIC ATTRIBUTES
# =============================================================================

def load_dealer_geo_mapping(session, logger):
    """
    Pulls AREA_OFFICE and ZONE from the dealer master view.
    The source column DEALER_CODE maps to PARENT_DEALER_CODE in the SOQ data
    (SOQ is already aggregated to parent-dealer level by Step 3).
    """
    dealer_data = (
        session.table(DEALER_MAPPING_TABLE)
        .select(col("DEALER_CODE"), col("AREA_OFFICE"), col("ZONE"))
        .to_pandas()
    )
    # Rename DEALER_CODE to PARENT_DEALER_CODE to align with the SOQ table's key
    dealer_data.columns = ['PARENT_DEALER_CODE', 'AREA_OFFICE', 'ZONE']
    logger.info("Dealer geo mapping loaded | Unique dealers: %s", dealer_data['PARENT_DEALER_CODE'].nunique())
    return dealer_data


# =============================================================================
# MAIN
# =============================================================================

def main(session: snowpark.Session):
    logger, log_stream = _setup_logger()
    logger.info("========== SOQ Step 5 Pipeline Started ==========")

    try:
        # Load SOQ output from Step 4
        soq_data, is_obd = load_soq_data(session, RUN_DATE, RUN_VERSION, logger)

        # Load dealer geographic dimension
        dealer_geo = load_dealer_geo_mapping(session, logger)

        # Left join: attach AREA_OFFICE and ZONE to every SOQ row.
        # Left join preserves all SOQ rows — dealers not in the master get nulls.
        soq_data = pd.merge(soq_data, dealer_geo, on='PARENT_DEALER_CODE', how='left')
        logger.info("Dealer geo joined | Rows after merge: %s", len(soq_data))

        # Re-attach IS_OBD (it was stripped before the merge to avoid alignment issues)
        soq_data['IS_OBD'] = is_obd

        # Save enriched output
        session.create_dataframe(soq_data).write.mode("append").save_as_table(OUTPUT_TABLE)
        logger.info("Output written to %s | Rows: %s", OUTPUT_TABLE, len(soq_data))

        logger.info("========== SOQ Step 5 Pipeline Complete ==========")

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        raise

    finally:
        print(log_stream.getvalue())

    return session.create_dataframe(soq_data)
