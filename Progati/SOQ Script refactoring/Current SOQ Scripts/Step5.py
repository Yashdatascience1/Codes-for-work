# The Snowpark package is required for Python Worksheets. 
# You can add more packages by selecting them using the Packages control and then importing them.

import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime

RUN_DATE=datetime.datetime.today().strftime('%Y%m%d')
#RUN_DATE = '20251102'
run_version=33

SOQ_TABLE=f'''SELECT * FROM MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_2  WHERE RUN_DATE='{RUN_DATE}' and run_version = {run_version} '''
#SOQ_TABLE=f'''SELECT * FROM MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_2_WEEKLY WHERE RUN_DATE='{RUN_DATE}' and run_version = {run_version} '''
DEALER_MAPPING_TABLE='ANALYTICS_DATABASE.ANALYTICS_SALES.VW_DEALER_MASTER'
OUTPUT_TABLE='MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_WITH_AO_VERSION_4'

def main(session: snowpark.Session): 
    soq_data=session.sql(SOQ_TABLE).to_pandas()
    is_obd=soq_data['IS_OBD'].tolist()
    soq_data.drop(['IS_OBD'],axis=1,inplace=True)
    dealer_data=session.table(DEALER_MAPPING_TABLE).to_pandas()
    dealer_data=dealer_data[['DEALER_CODE','AREA_OFFICE','ZONE']]
    dealer_data.columns=['PARENT_DEALER_CODE','AREA_OFFICE','ZONE']
    soq_data=pd.merge(soq_data,dealer_data,on=['PARENT_DEALER_CODE'],how='left')
    soq_data['IS_OBD']=is_obd
    agg_data_sp=session.create_dataframe(soq_data)
    agg_data_sp.write.mode("append").save_as_table(OUTPUT_TABLE)
    return agg_data_sp
    
