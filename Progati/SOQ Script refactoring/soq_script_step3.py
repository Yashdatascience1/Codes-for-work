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
LOOKBACK_MONTHS = 3
CUSTOMER_TYPE           = ('Individual',)
IS_OBD                  = True

def get_shape(df):
    return (df.count(), len(df.columns))
	
def get_null_counts(df):
    null_exprs = [
        F.count(F.iff(F.col(c).is_null(), F.lit(1), F.lit(None))).alias(f"NUMBER_OF_NULL_VALUES_IN_{c}")
        for c in df.columns
    ]

forecast = session.table(TFT_FORECAST_TABLE)

#Data quality check
forecast.group_by(F.col("PARENT_DEALER_CODE_MODEL_FAMILY"),F.col("MONTH_OF_SALE")).agg(F.count_distinct(F.col("PREDICTED_SALES_TFT")).alias("UNIQUE_COUNT_OF_NET_SALES")).order_by(F.col("UNIQUE_COUNT_OF_NET_SALES").desc()).show()

#Convert MONTH_OF_SALE to Date type
forecast = forecast.with_column("MONTH_OF_SALE",F.to_date(F.col("MONTH_OF_SALE")))

#Fetching the dealer code
forecast = forecast.with_column("PARENT_DEALER_CODE",
                                F.trim(F.split(F.col("PARENT_DEALER_CODE_MODEL_FAMILY"),
                                F.lit('<>'))[0]))

#Fetching the part after the dealer code
forecast = forecast.with_column("UNIQUE_FAMILY_CODE",F.substr(F.col("PARENT_DEALER_CODE_MODEL_FAMILY"),
                                (F.charindex(F.lit('<>'),F.col("PARENT_DEALER_CODE_MODEL_FAMILY")))+F.lit(2)
                                ))

#Load the parent dealer view
parent_dealer_view = session.table(PARENT_DEALER_VIEW)

#Checking if parent_dealer_view snowpark dataframe has null values
parent_dealer_view.select(F.sum(F.col("X_DEALER_CODE_HIER").is_null().cast("int")).alias("NUMBER_OF_NULL_VALUES")).show()

columns_count = len(parent_dealer_view.columns)
rows_count = parent_dealer_view.count()

print(f"Shape of parent_dealer_view before removing nulls: ({rows_count}, {columns_count})")

#Filtering out null values from the X_DEALER_CODE_HIER column
parent_dealer_view = parent_dealer_view.filter(~F.col("X_DEALER_CODE_HIER").is_null())

columns_count = len(parent_dealer_view.columns)
rows_count = parent_dealer_view.count()

print(f"Shape of parent_dealer_view after removing nulls: ({rows_count}, {columns_count})")

#Extract parent dealer code from the snowpark dataframe
parent_dealer_view = parent_dealer_view.with_column("PARENT_DEALER_CODE",F.trim(F.split(F.col("PAR_ORG_NAME"),F.lit('-'))[0]))

#Select only PARENT_DEALER_CODE and DEALER_CODE
parent_dealer_view = parent_dealer_view.select(F.col("PARENT_DEALER_CODE"),F.col("X_DEALER_CODE_HIER").alias("DEALER_CODE"))

get_shape(parent_dealer_view)

print(f"Before taking distinct : Shape of the dataframe is {get_shape(parent_dealer_view)}")

parent_dealer_view = parent_dealer_view.distinct()

print(f"After taking distinct : Shape of the dataframe is {get_shape(parent_dealer_view)}")

parent_dealer_view.select(F.sum(F.col("PARENT_DEALER_CODE").is_null().cast('int')).alias("NUMBER_OF_NULLS_IN_PARENT_DEALER_CODE")).show()

sku_supercedence = (
        session.table(SKU_SUPERCEDENCE_TABLE)
    )



	

#Filter for only active SKUs
active_skus = sku_supercedence.filter(F.col('SKUSTATUS') == F.lit('active'))

#Remove " from column names
for old_col in active_skus.columns:
    new_col = old_col.replace('"','')
    active_skus=active_skus.rename(old_col,new_col)
	

num_active = (
        active_skus
        .group_by('UNIQUE FAMILY CODE')
        .agg(F.count_distinct('SKU').alias('NUM_ACTIVE_SKUS'))
    )

num_active.select(F.count_distinct(F.col("UNIQUE FAMILY CODE")).alias("NUMBER_OF_UNIQUE_FAMILY_CODES")).show()

num_active.select(F.sum(F.col("NUM_ACTIVE_SKUS")).alias("TOTAL_ACTIVE_SKUS")).show()


