import sys 
sys.path.append(r"C:\Users\g0004878\Desktop\My codes\Frequently used codes\Frequently-used-codes")
sys.path.append(r"C:\Users\g0004878\Desktop\Projects in FY25-26\utils_files")
import Snowflake_configuration
snowflake_conn_prop = Snowflake_configuration.ds1_role_json
from snowflake.snowpark.session import Session
import pandas as pd
pd.set_option('display.max_columns',None)

session = Session.builder.configs(snowflake_conn_prop).create()

import snowflake.snowpark as snowpark
from snowflake.snowpark import functions as F
from snowflake.snowpark.window import Window

TFT_FORECAST_TABLE      = 'MOP_DATABASE.SOQ.PREDICTIONS_BY_TFT_JAN_26_TO_APR_26'
OUTPUT_TABLE            = 'MOP_DATABASE.SOQ.TFT_PREDICTIONS_DISAGGREGATED_SKU_LEVEL'
ECR_TABLE               = 'ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS'
SKU_SUPERCEDENCE_TABLE  = 'MOP_DATABASE.SOQ.SKU_SUPERCEDENCE_MODEL_FAMILY_FEB_2025_UPDATED_V2'
OBD_MAPPING_TABLE       = 'MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW'
PARENT_DEALER_VIEW      = 'FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH'
OUTPUT_TABLE            = 'MOP_DATABASE.SOQ.DEALER_SKU_DISAGGREGATION_RESULTS'
CUSTOMER_TYPE           = ('Individual',)
IS_OBD                  = True
LOOKBACK_MONTHS         = 3

def get_shape(df):
    return (df.count(), len(df.columns))

def get_null_counts(df):
    null_exprs = [
        F.count(F.iff(F.col(c).is_null(), F.lit(1), F.lit(None))).alias(f"NUMBER_OF_NULL_VALUES_IN_{c}")
        for c in df.columns
    ]

forecast = session.table(TFT_FORECAST_TABLE)

forecast.group_by(F.col("PARENT_DEALER_CODE_MODEL_FAMILY"),F.col("MONTH_OF_SALE")).agg(F.count_distinct(F.col("PREDICTED_SALES_TFT")).alias("UNIQUE_COUNT_OF_NET_SALES")).order_by(F.col("UNIQUE_COUNT_OF_NET_SALES").desc()).show()

#Convert MONTH_OF_SALE to Date type
forecast = forecast.with_column("MONTH_OF_SALE",F.to_date(F.col("MONTH_OF_SALE")))

#Fetching the dealer code
#Prompt : In the logger - show me the number of unique dealer codes - Specify the table name
forecast = forecast.with_column("PARENT_DEALER_CODE",
                                F.trim(F.split(F.col("PARENT_DEALER_CODE_MODEL_FAMILY"),
                                F.lit('<>'))[0]))

#Fetching the part after the dealer code
#Prompt : In the logger - show me the number of unique family codes - Specify the table name
forecast = forecast.with_column("UNIQUE_FAMILY_CODE",F.substr(F.col("PARENT_DEALER_CODE_MODEL_FAMILY"),
                                (F.charindex(F.lit('<>'),F.col("PARENT_DEALER_CODE_MODEL_FAMILY")))+F.lit(2)
                                ))


parent_dealer_view = session.table(PARENT_DEALER_VIEW)

parent_dealer_view.select(F.sum(F.col("X_DEALER_CODE_HIER").is_null().cast("int")).alias("NUMBER_OF_NULL_VALUES")).show()

parent_dealer_view = parent_dealer_view.filter(~F.col("X_DEALER_CODE_HIER").is_null())

columns_count = len(parent_dealer_view.columns)
rows_count = parent_dealer_view.count()

#Extract parent dealer code from the snowpark dataframe
parent_dealer_view = parent_dealer_view.with_column("PARENT_DEALER_CODE",F.trim(F.split(F.col("PAR_ORG_NAME"),F.lit('-'))[0]))

#Select only PARENT_DEALER_CODE and DEALER_CODE
parent_dealer_view = parent_dealer_view.select(F.col("PARENT_DEALER_CODE"),F.col("X_DEALER_CODE_HIER").alias("DEALER_CODE"))

parent_dealer_view.select(F.sum(F.col("PARENT_DEALER_CODE").is_null().cast('int')).alias("NUMBER_OF_NULLS_IN_PARENT_DEALER_CODE")).show()

sku_supercedence = (
        session.table(SKU_SUPERCEDENCE_TABLE)
        .drop('UPDATED_ON')
    )

sku_supercedence = sku_supercedence.with_column('UPDATED_ON',F.to_date(F.col('UPDATED_ON').cast("string"),F.lit('YYYYMMDD')))

#Prompt in the logger - show me the name of the SKU_SUPERCEDENCE_TABLE 
#Also mention the latest date when SKU status was updated
latest_update_date_in_sku_supercedence=sku_supercedence.select(F.max("UPDATED_ON")).collect()[0][0]

sku_supercedence = sku_supercedence.drop("UPDATED_ON")

active_skus = sku_supercedence.filter(F.col('SKUSTATUS') == F.lit('active'))

for old_col in active_skus.columns:
    new_col = old_col.replace('"','')
    active_skus=active_skus.rename(old_col,new_col)

num_active = (
        active_skus
        .group_by('UNIQUE FAMILY CODE')
        .agg(F.count_distinct('SKU').alias('NUM_ACTIVE_SKUS'))
    )

#Prompt : In the logger - show me the number of unique family codes - Specify the table name
num_active.select(F.count_distinct(F.col("UNIQUE FAMILY CODE")).alias("NUMBER_OF_UNIQUE_FAMILY_CODES")).show()

#Prompt : In the logger - show me the number of unique SKUs - Specify the table name
num_active.select(F.sum(F.col("NUM_ACTIVE_SKUS")).alias("TOTAL_ACTIVE_SKUS")).show()

forecast_month_bounds = (
    session.table(TFT_FORECAST_TABLE)
    .select(
        F.min('MONTH_OF_SALE').alias('MIN_FORECAST_MONTH'),
        F.max('MONTH_OF_SALE').alias('MAX_FORECAST_MONTH')
    )
    .collect()[0]
)

earliest_forecast_month = forecast_month_bounds['MIN_FORECAST_MONTH']
latest_forecast_month   = forecast_month_bounds['MAX_FORECAST_MONTH']

# Lookback start = LOOKBACK_MONTHS before the earliest forecast month
lookback_start = F.add_months(F.lit(earliest_forecast_month), -LOOKBACK_MONTHS)

ecr_raw = (
    session.table(ECR_TABLE)
    .filter(F.col('X_CUSTOMER_TYPE').isin(list(CUSTOMER_TYPE)))
    .filter(F.col('CAL_DATE') >= lookback_start)
    .filter(F.col('CAL_DATE') <  F.lit(latest_forecast_month))
    .with_column('CAL_DATE', F.to_date(F.col('CAL_DATE')))
    .with_column(
        'NET_SALES',
        F.col('INVOICED_SALES') + F.col('CANCELLED_SALES') + F.col('RETURNED_SALES')
    )
)

sku_supercedence=sku_supercedence.rename('UNIQUE FAMILY CODE','UNIQUE_FAMILY_CODE')

ecr = (
    ecr_raw
    .join(sku_supercedence.select('SKU', 'MODEL', 'UNIQUE_FAMILY_CODE'),
            on=['SKU', 'MODEL'], how='left')
    .join(parent_dealer_view, on='DEALER_CODE', how='left')
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

forecast_months = forecast.select('MONTH_OF_SALE').distinct()

forecast_months = forecast.select('MONTH_OF_SALE').distinct()

ecr_with_month = ecr.join(forecast_months, how='cross')

ecr_windowed = ecr_with_month.filter(
    (F.col('CAL_DATE') >= F.add_months(F.col('MONTH_OF_SALE'), F.lit(-3))) &
    (F.col('CAL_DATE') <  F.col('MONTH_OF_SALE'))

dealer_family_sku_sales = (
        ecr_windowed
        .group_by(['MONTH_OF_SALE', 'PARENT_DEALER_CODE', 'UNIQUE_FAMILY_CODE', 'SKU'])
        .agg(F.sum('NET_SALES').alias('DEALER_SKU_SALES'))
    )


def removing_double_quotes_from_column_names(df):
    for old_col in df.columns:
        new_col = old_col.replace('"', '')
        df = df.with_column_renamed(old_col, new_col) 
    return df 

num_active = removing_double_quotes_from_column_names(num_active)
active_skus = removing_double_quotes_from_column_names(active_skus)

active_skus = active_skus.with_column_renamed('UNIQUE FAMILY CODE','UNIQUE_FAMILY_CODE')

num_active = num_active.with_column_renamed('UNIQUE FAMILY CODE','UNIQUE_FAMILY_CODE')

forecast_sku = (
        forecast
        .join(
            active_skus.select(F.col('UNIQUE_FAMILY_CODE'), 'SKU'),
            on='UNIQUE_FAMILY_CODE',
            how='left'
        )
        .join(
            num_active,
            on='UNIQUE_FAMILY_CODE', 
            how='left'
        )
    )

disaggregated = (
    forecast_sku
    .join(
        dealer_family_sku_sales.select(
            'MONTH_OF_SALE', 'PARENT_DEALER_CODE', 'UNIQUE_FAMILY_CODE',
            'SKU', 'DEALER_SKU_SALES'
        ),
        on=['MONTH_OF_SALE', 'PARENT_DEALER_CODE', 'UNIQUE_FAMILY_CODE', 'SKU'],
        how='left'
    )
    .with_column('DEALER_SKU_SALES', F.coalesce(F.col('DEALER_SKU_SALES'), F.lit(0.0)))
)

# NEW STEP: Calculate FAMILY_TOTAL_SALES here so it only sums ACTIVE SKUs!
family_window = Window.partition_by('MONTH_OF_SALE', 'PARENT_DEALER_CODE', 'UNIQUE_FAMILY_CODE')

disaggregated = disaggregated.with_column(
    'FAMILY_TOTAL_SALES',
    F.sum('DEALER_SKU_SALES').over(family_window)


disaggregated = disaggregated.with_column(
        'PERCENT_PROPORTION',
        F.when(
            # No history at all for this family → equal split
            F.col('FAMILY_TOTAL_SALES') == F.lit(0),
            F.lit(1.0) / F.col('NUM_ACTIVE_SKUS')
        ).when(
            # Family has history but this SKU has zero → gets 0
            F.col('DEALER_SKU_SALES') == F.lit(0),
            F.lit(0.0)
        ).otherwise(
            F.col('DEALER_SKU_SALES') / F.col('FAMILY_TOTAL_SALES')
        )
    )

disaggregated = disaggregated.with_column(
    'PREDICTED_SALES_SKU_TFT',
    F.when(
        F.col('NUM_ACTIVE_SKUS') == F.lit(1),
        F.col('PREDICTED_SALES_TFT')                              # single SKU — no split needed
    ).otherwise(
        F.col('PERCENT_PROPORTION') * F.col('PREDICTED_SALES_TFT')
    )
)

disaggregated = disaggregated.with_column(
    'NO_HISTORY_FLAG',
    F.when(F.col('FAMILY_TOTAL_SALES') == F.lit(0), F.lit(True))
        .otherwise(F.lit(False))
)

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
    'FAMILY_TOTAL_SALES',
    'NO_HISTORY_FLAG'
)

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

#Prompt mention the 3 metrics in the logger 

output.write.mode("overwrite").save_as_table(OUTPUT_TABLE)


