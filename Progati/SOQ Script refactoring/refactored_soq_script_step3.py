import snowflake.snowpark as snowpark
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window

# ── CONFIG ────────────────────────────────────────────────────────────────────
TFT_FORECAST_TABLE      = 'MOP_DATABASE.SOQ.PREDICTIONS_BY_TFT_JAN_26_TO_APR_26'
OUTPUT_TABLE            = 'MOP_DATABASE.SOQ.TFT_PREDICTIONS_DISAGGREGATED_SKU_LEVEL'
ECR_TABLE               = 'ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS'
SKU_SUPERCEDENCE_TABLE  = 'MOP_DATABASE.SOQ.SKU_SUPERCEDENCE_MODEL_FAMILY_FEB_2025_UPDATED_V2'
OBD_MAPPING_TABLE       = 'MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW'
PARENT_DEALER_VIEW      = 'FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH'
CUSTOMER_TYPE           = ('Individual',)
IS_OBD                  = True
# ─────────────────────────────────────────────────────────────────────────────


def main(session: snowpark.Session):

    # ── Step 1: Load forecast table, parse dealer and family from key column ──
    forecast = (
        session.table(TFT_FORECAST_TABLE)
        .with_column('MONTH_OF_SALE', F.to_date(F.col('MONTH_OF_SALE')))
        .with_column(
            'PARENT_DEALER_CODE',
            F.split(F.col('PARENT_DEALER_CODE_MODEL_FAMILY'), F.lit('<>'))[0]
        )
        .with_column(
            # UNIQUE FAMILY CODE = everything after the first <> joined back
            # e.g. 12015<>PASSION<>DRUM → PASSION<>DRUM
            'UNIQUE_FAMILY_CODE',
            F.regexp_replace(
                F.col('PARENT_DEALER_CODE_MODEL_FAMILY'),
                F.lit('^[^<>]+<>'),   # strip leading DEALER_CODE<>
                F.lit('')
            )
        )
    )

    # ── Step 2: Parent dealer mapping ─────────────────────────────────────────
    parent_dealer = (
        session.table(PARENT_DEALER_VIEW)
        .filter(F.col('X_DEALER_CODE_HIER').is_not_null())
        .select(
            F.col('X_DEALER_CODE_HIER').alias('DEALER_CODE'),
            F.trim(F.split(F.col('PAR_ORG_NAME'), F.lit('-'))[0]).alias('PARENT_DEALER_CODE')
        )
        .distinct()
    )

    # ── Step 3: SKU supercedence — active SKUs only ───────────────────────────
    sku_supercedence = (
        session.table(SKU_SUPERCEDENCE_TABLE)
        .drop('UPDATED_ON')
    )

    active_skus = sku_supercedence.filter(F.col('SKUSTATUS') == F.lit('active'))

    # Number of active SKUs per family
    num_active = (
        active_skus
        .group_by('UNIQUE FAMILY CODE')
        .agg(F.count_distinct('SKU').alias('NUM_ACTIVE_SKUS'))
    )

    # ── Step 4: ECR — pull all history needed across all forecast months ───────
    # Earliest lookback = 3 months before earliest forecast month (Oct 2025)
    # Latest lookback   = last day before latest forecast month  (Mar 2026)
    # Pulling one wide window; per-month filtering happens in Step 6 via join
    ecr_raw = (
        session.table(ECR_TABLE)
        .filter(F.col('X_CUSTOMER_TYPE').isin(list(CUSTOMER_TYPE)))
        .filter(F.col('CAL_DATE') >= F.lit('2025-10-01'))   # 3M before Jan 2026
        .filter(F.col('CAL_DATE') <  F.lit('2026-04-01'))   # up to last forecast month
        .with_column('CAL_DATE', F.to_date(F.col('CAL_DATE')))
        .with_column(
            'NET_SALES',
            F.col('INVOICED_SALES') + F.col('CANCELLED_SALES') + F.col('RETURNED_SALES')
        )
    )

    # ── Step 5: Map ECR → parent dealer + supercedence ────────────────────────
    ecr = (
        ecr_raw
        .join(sku_supercedence.select('SKU', 'MODEL', 'UNIQUE FAMILY CODE'),
              on=['SKU', 'MODEL'], how='left')
        .join(parent_dealer, on='DEALER_CODE', how='left')
    )

    # OBD mapping — replace SKU with current OBD variant
    if IS_OBD:
        obd = (
            session.table(OBD_MAPPING_TABLE)
            .select(
                F.col('PREVIOUS_OBD_SKU').alias('SKU'),
                F.col('CURRENT_OBD_SKU')
            )
        )
        ecr = (
            ecr
            .join(obd, on='SKU', how='left')
            .with_column(
                'SKU',
                F.coalesce(F.col('CURRENT_OBD_SKU'), F.col('SKU'))
            )
            .drop('CURRENT_OBD_SKU')
        )

    # ── Step 6: Cross-join forecast months to ECR, apply rolling 3M filter ────
    # This is the Snowpark equivalent of the Python loop over forecast months
    forecast_months = forecast.select('MONTH_OF_SALE').distinct()

    ecr_with_month = ecr.join(forecast_months, how='cross')

    ecr_windowed = ecr_with_month.filter(
        (F.col('CAL_DATE') >= F.add_months(F.col('MONTH_OF_SALE'), F.lit(-3))) &
        (F.col('CAL_DATE') <  F.col('MONTH_OF_SALE'))
    )

    # ── Step 7: SKU-level sales per dealer-family per forecast month ──────────
    dealer_family_sku_sales = (
        ecr_windowed
        .group_by(['MONTH_OF_SALE', 'PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE', 'SKU'])
        .agg(F.sum('NET_SALES').alias('DEALER_SKU_SALES'))
    )

    # Family-level totals (sum of active SKU sales = denominator for weights)
    family_window = Window.partition_by(
        'MONTH_OF_SALE', 'PARENT_DEALER_CODE', 'UNIQUE FAMILY CODE'
    )

    dealer_family_sku_sales = dealer_family_sku_sales.with_column(
        'TOTAL_DEALER_ACTIVE_SKU_SALES',
        F.sum('DEALER_SKU_SALES').over(family_window)
    )

    dealer_family_sku_sales = dealer_family_sku_sales.with_column(
        'DEALER_FAMILY_CODE_NET_SALES',
        F.sum('DEALER_SKU_SALES').over(family_window)
    )

    # ── Step 8: Expand forecast to SKU level via active SKU supercedence ──────
    forecast_sku = (
        forecast
        .join(
            active_skus.select(
                F.col('UNIQUE FAMILY CODE').alias('UNIQUE_FAMILY_CODE'),
                'SKU'
            ),
            on='UNIQUE_FAMILY_CODE',
            how='left'
        )
        .join(num_active.rename(F.col('UNIQUE FAMILY CODE'), 'UNIQUE_FAMILY_CODE'),
              on='UNIQUE_FAMILY_CODE', how='left')
    )

    # ── Step 9: Join SKU weights onto expanded forecast ───────────────────────
    disaggregated = (
        forecast_sku
        .join(
            dealer_family_sku_sales.select(
                'MONTH_OF_SALE', 'PARENT_DEALER_CODE',
                F.col('UNIQUE FAMILY CODE').alias('UNIQUE_FAMILY_CODE'),
                'SKU', 'DEALER_SKU_SALES',
                'TOTAL_DEALER_ACTIVE_SKU_SALES',
                'DEALER_FAMILY_CODE_NET_SALES'
            ),
            on=['MONTH_OF_SALE', 'PARENT_DEALER_CODE', 'UNIQUE_FAMILY_CODE', 'SKU'],
            how='left'
        )
        .with_column('DEALER_SKU_SALES',
                     F.coalesce(F.col('DEALER_SKU_SALES'), F.lit(0)))
        .with_column('TOTAL_DEALER_ACTIVE_SKU_SALES',
                     F.coalesce(F.col('TOTAL_DEALER_ACTIVE_SKU_SALES'), F.lit(0)))
        .with_column('DEALER_FAMILY_CODE_NET_SALES',
                     F.coalesce(F.col('DEALER_FAMILY_CODE_NET_SALES'), F.lit(0)))
    )

    # ── Step 10: Compute SKU weight (mirrors percentsku logic exactly) ─────────
    disaggregated = disaggregated.with_column(
        'PERCENT_PROPORTION',
        F.when(
            # No history at all for this family → equal split
            F.col('TOTAL_DEALER_ACTIVE_SKU_SALES') == F.lit(0),
            F.lit(1.0) / F.col('NUM_ACTIVE_SKUS')
        ).when(
            # Family has history but this SKU has zero → gets 0
            F.col('DEALER_SKU_SALES') == F.lit(0),
            F.lit(0.0)
        ).otherwise(
            F.col('DEALER_SKU_SALES') / F.col('TOTAL_DEALER_ACTIVE_SKU_SALES')
        )
    )

    # ── Step 11: Disaggregate predicted sales ─────────────────────────────────
    disaggregated = disaggregated.with_column(
        'PREDICTED_SALES_SKU_TFT',
        F.when(
            F.col('NUM_ACTIVE_SKUS') == F.lit(1),
            F.col('PREDICTED_SALES_TFT')                              # single SKU — no split needed
        ).otherwise(
            F.col('PERCENT_PROPORTION') * F.col('PREDICTED_SALES_TFT')
        )
    )

    # ── Step 12: Flag families with no ECR history ────────────────────────────
    disaggregated = disaggregated.with_column(
        'NO_HISTORY_FLAG',
        F.when(F.col('DEALER_FAMILY_CODE_NET_SALES') == F.lit(0), F.lit(True))
         .otherwise(F.lit(False))
    )

    # ── Step 13: Final output columns ─────────────────────────────────────────
    output = disaggregated.select(
        'MONTH_OF_SALE',
        'PARENT_DEALER_CODE_MODEL_FAMILY',
        'PARENT_DEALER_CODE',
        'UNIQUE_FAMILY_CODE',
        'SKU',
        'NUM_ACTIVE_SKUS',
        'PREDICTED_SALES_TFT',
        F.round('PERCENT_PROPORTION', 5).alias('PERCENT_PROPORTION'),
        F.round('PREDICTED_SALES_SKU_TFT', 4).alias('PREDICTED_SALES_SKU_TFT'),
        'DEALER_SKU_SALES',
        'DEALER_FAMILY_CODE_NET_SALES',
        'NO_HISTORY_FLAG'
    )

    # ── Step 14: Sanity check before writing ──────────────────────────────────
    check = (
        output.filter(F.col('NO_HISTORY_FLAG') == F.lit(False))
        .group_by(['MONTH_OF_SALE', 'PARENT_DEALER_CODE_MODEL_FAMILY'])
        .agg(
            F.max('PREDICTED_SALES_TFT').alias('ORIGINAL'),
            F.sum('PREDICTED_SALES_SKU_TFT').alias('REAGGREGATED')
        )
        .with_column('DIFF', F.abs(F.col('ORIGINAL') - F.col('REAGGREGATED')))
    )

    max_diff         = check.agg(F.max('DIFF').alias('MAX_DIFF')).collect()[0]['MAX_DIFF']
    no_history_count = output.filter(F.col('NO_HISTORY_FLAG') == F.lit(True)).count()
    total_rows       = output.count()

    print(f"Total output rows        : {total_rows}")
    print(f"NO_HISTORY_FLAG rows     : {no_history_count}")
    print(f"Max reaggregation diff   : {max_diff:.6f}")

    # ── Step 15: Write to Snowflake ───────────────────────────────────────────
    output.write.mode("overwrite").save_as_table(OUTPUT_TABLE)
    print(f"Output written to        : {OUTPUT_TABLE}")

    return session.table(OUTPUT_TABLE)


