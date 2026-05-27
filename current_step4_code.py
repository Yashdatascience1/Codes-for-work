import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import math
import numpy as np
import pandas as pd
import datetime


MONTHS=['2026-06-01'] ## 

STOCK_DATE_TYPE=["first"]#,"mid"]   

RUN_VERSION=33

ABC={"A":30,"B":25,"C":20}

RUN_DATE=datetime.datetime.today().strftime('%Y%m%d')


Z_SCORE= {95:1.65,90:1.28,80:0.85,85:1.04,99:2.33}

STR_ABC= ','.join([f"{key} {value}" for key, value in ABC.items()])

BASE_SOQ_TABLE="MOP_DATABASE.SOQ.SOQ_BASE_TABLE_FINAL_CONCATENATED"

DEMAND_VARIABILITY_TABLE='MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_FINAL_VERSION'

DEMAND_VARIABILITY_FAMILY_TABLE='MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_MODEL_FAMILY_FINAL_VERSION'



SOQ_TABLE='MOP_DATABASE.SOQ.SOQ_DATA_FINAL_VERSION_2'

END_JOURNEY_QUERY='''
SELECT SKU, RECOMMEND_END_OF_JOURNEY FROM MOP_DATABASE.SOQ.END_OF_JOURNEY_RECOMMENDATION WHERE RUN_DATE=(SELECT MAX(RUN_DATE) FROM MOP_DATABASE.SOQ.END_OF_JOURNEY_RECOMMENDATION)
'''

IS_OBD=True
if IS_OBD:
    obd_flag='Y'
else:
    obd_flag='N'

def getABC(perc):
    if perc < 70:
        return 'A'
    if perc > 90:
        return 'C'
    else:
        return 'B'
def calculateSOQ(session,planning_month,date_period,service_level=95,run_date=RUN_DATE,sku_demand_variability=True,run_version=RUN_VERSION):
    soq_base_query=f'''
    SELECT * FROM {BASE_SOQ_TABLE} WHERE PLANNING_MONTH='{planning_month}'
    AND STOCK_DATE_PERIOD='{date_period}' AND RUN_DATE={run_date} AND RUN_VERSION={run_version} AND IS_OBD='{obd_flag}'
    '''
    print(soq_base_query)
    soq_data=session.sql(soq_base_query).to_pandas()
    soq_data.drop(['IS_OBD'],axis=1,inplace=True)
    print(soq_data.head())
    if sku_demand_variability==True:
        demand_query=f'''
        SELECT * FROM {DEMAND_VARIABILITY_TABLE} WHERE PLANNING_MONTH='{planning_month}' AND RUN_DATE={run_date} AND RUN_VERSION={run_version}
        AND IS_OBD='{obd_flag}'
        '''
        
        demand_variability=session.sql(demand_query).to_pandas()
        demand_variability.drop(['IS_OBD'],axis=1,inplace=True)
        
        data=pd.merge(soq_data,demand_variability,on=['PARENT_DEALER_CODE','UNIQUE FAMILY CODE','SKU','PLANNING_MONTH'],how="left")
        data.loc[pd.isnull(data['DEMAND_VARIABILITY']),'DEMAND_VARIABILITY']=1
        data['DEMAND_VARIABILITY_TYPE']="SKU_BASED"
        
    else:
        demand_query=f'''
        SELECT * FROM {DEMAND_VARIABILITY_FAMILY_TABLE} WHERE PLANNING_MONTH='{planning_month}' AND RUN_DATE={run_date}  AND RUN_VERSION={run_version}
        AND IS_OBD='{obd_flag}'
        '''
        demand_variability=session.sql(demand_query).to_pandas()
        demand_variability.drop(['IS_OBD'],axis=1,inplace=True)
        
        data=pd.merge(soq_data,demand_variability,on=['PARENT_DEALER_CODE','UNIQUE FAMILY CODE','PLANNING_MONTH'],how="left")
        data.loc[pd.isnull(data['DEMAND_VARIABILITY']),'DEMAND_VARIABILITY']=1

        data['DEMAND_VARIABILITY_TYPE']="MODEL_SKU_FAMILY_BASED"
        


    ### Calculate ABC
    sales_data=data[['PARENT_DEALER_CODE','UNIQUE FAMILY CODE','PREDICTED_SALES']].drop_duplicates()
    sales_data.shape
    dealer_sales=sales_data.groupby(['PARENT_DEALER_CODE'])['PREDICTED_SALES'].sum().reset_index().rename(columns={'PREDICTED_SALES':'DEALER_PREDICTED_SALES'})
    sales_data=pd.merge(sales_data,dealer_sales,on=['PARENT_DEALER_CODE'],how="left")
    sales_data.head()
    sales_data['PERC_SALES']=(sales_data['PREDICTED_SALES']/sales_data['DEALER_PREDICTED_SALES'])*100
    sales_data.loc[pd.isnull(sales_data['PERC_SALES']),'PERC_SALES']=0
    sales_data=sales_data.sort_values(by=['PARENT_DEALER_CODE','PERC_SALES'],ascending=False)
    sales_data['CUMULATIVE_PERCENT_SALES'] = sales_data.groupby('PARENT_DEALER_CODE')['PERC_SALES'].cumsum()
    sales_data.head()
    sales_data['ABC']=sales_data['CUMULATIVE_PERCENT_SALES'].apply(lambda x:getABC(x))
    sales_data=sales_data[['PARENT_DEALER_CODE','UNIQUE FAMILY CODE','ABC']]
    data=pd.merge(data,sales_data,on=['PARENT_DEALER_CODE','UNIQUE FAMILY CODE'],how="left")
    data['SAFETY_STOCK_DAYS']=data['ABC'].apply(lambda x:ABC[x] if x in ABC else 15)
    null_lead_time=session.create_dataframe(data[pd.isnull(data['MAX_LEAD_TIME'])])
    null_lead_time.write.mode("overwrite").save_as_table("MOP_DATABASE.SOQ.NULL_LEAD_TIME")
    data=data[~pd.isnull(data['MAX_LEAD_TIME'])]
    data['MAX_LEAD_TIME_STOCK']=data.apply(lambda row:(row['PREDICTED_SALES_SKU']*row['MAX_LEAD_TIME'])/30,axis=1)
    data['MIN_LEAD_TIME_STOCK']=data.apply(lambda row:(row['PREDICTED_SALES_SKU']*row['MIN_LEAD_TIME'])/30,axis=1)
    data['AVG_LEAD_TIME_STOCK']=data.apply(lambda row:(row['PREDICTED_SALES_SKU']*row['AVG_LEAD_TIME'])/30,axis=1)

    data['MAX_LEAD_TIME_STOCK']=data['MAX_LEAD_TIME_STOCK'].apply(lambda x:math.ceil(x) )
    data['MIN_LEAD_TIME_STOCK']=data['MIN_LEAD_TIME_STOCK'].apply(lambda x:math.ceil(x))
    data['AVG_LEAD_TIME_STOCK']=data['AVG_LEAD_TIME_STOCK'].apply(lambda x:math.ceil(x))

    data['SAFETY_STOCK']= data['DEMAND_VARIABILITY'] * np.sqrt(data['SAFETY_STOCK_DAYS']/30) 
    data['SAFETY_STOCK']= data['SAFETY_STOCK'].apply(lambda x:math.ceil(x * Z_SCORE[service_level]))

  

    data['SAFETY_STOCK'] =  np.minimum(data['SAFETY_STOCK'].fillna(0),(data['PREDICTED_SALES_SKU'].fillna(0) * 3))
         
    
    data.loc[pd.isnull(data['STK_AS_ON_DATE']),'STK_AS_ON_DATE']=0
    data['MAX_REORDER_STOCK']=data['SAFETY_STOCK']+data['MAX_LEAD_TIME_STOCK']
    data['MAX_TOTAL_STOCK_SKU']=data['PREDICTED_SALES_SKU']+data['MAX_REORDER_STOCK']
    data['MAX_Suggested_Stock_SKU']=data['MAX_TOTAL_STOCK_SKU'] - data['STK_AS_ON_DATE']
    data['MAX_Adjusted_Monthly_Order']=data['MAX_Suggested_Stock_SKU'].apply(lambda x: 0 if x<0 else x)

    data['MIN_REORDER_STOCK']=data['SAFETY_STOCK']+data['MIN_LEAD_TIME_STOCK']
    data['MIN_TOTAL_STOCK_SKU']=data['PREDICTED_SALES_SKU']+data['MIN_REORDER_STOCK']
    data['MIN_Suggested_Stock_SKU']=data['MIN_TOTAL_STOCK_SKU'] - data['STK_AS_ON_DATE']
    data['MIN_Adjusted_Monthly_Order']=data['MIN_Suggested_Stock_SKU'].apply(lambda x: 0 if x<0 else x)


    data['AVG_REORDER_STOCK']=data['SAFETY_STOCK']+data['AVG_LEAD_TIME_STOCK']
    data['AVG_TOTAL_STOCK_SKU']=data['PREDICTED_SALES_SKU']+data['AVG_REORDER_STOCK']
    data['AVG_Suggested_Stock_SKU']=data['AVG_TOTAL_STOCK_SKU'] - data['STK_AS_ON_DATE']
    data['AVG_Adjusted_Monthly_Order']=data['AVG_Suggested_Stock_SKU'].apply(lambda x: 0 if x<0 else x)
    data['ABC']=STR_ABC
    data['Z_SCORE']=Z_SCORE[service_level]
    data['SERVICE_LEVEL']=service_level


    data['AVG_SOQ_APPROACH_2']=data['AVG_REORDER_STOCK']-data['STK_AS_ON_DATE']
    data['MAX_SOQ_APPROACH_2']=data['MAX_REORDER_STOCK']-data['STK_AS_ON_DATE']
    data['MIN_SOQ_APPROACH_2']=data['MIN_REORDER_STOCK']-data['STK_AS_ON_DATE']

    data['AVG_Adjusted_Monthly_Order_APPROACH_2']=data['AVG_SOQ_APPROACH_2'].apply(lambda x: 0 if x<0 else x)
    data['MAX_Adjusted_Monthly_Order_APPROACH_2']=data['MAX_SOQ_APPROACH_2'].apply(lambda x: 0 if x<0 else x)
    data['MIN_Adjusted_Monthly_Order_APPROACH_2']=data['MIN_SOQ_APPROACH_2'].apply(lambda x: 0 if x<0 else x)

    data['RUN_DATE']=run_date

    end_journey=session.sql(END_JOURNEY_QUERY).to_pandas()

    data=pd.merge(data,end_journey,how="left",on='SKU')
    data['RUN_VERSION']=run_version
    data['IS_OBD']=obd_flag
    agg_data_sp=session.create_dataframe(data)
    print(agg_data_sp)
    agg_data_sp.write.mode("append").save_as_table(SOQ_TABLE)
    
    

def main(session: snowpark.Session): 
    for months in MONTHS:
        for types in STOCK_DATE_TYPE:
            calculateSOQ(session,months,types,95,RUN_DATE,True, RUN_VERSION)
            calculateSOQ(session,months,types,90,RUN_DATE,True, RUN_VERSION)
            calculateSOQ(session,months,types,85,RUN_DATE,True, RUN_VERSION)
            calculateSOQ(session,months,types,99,RUN_DATE,True, RUN_VERSION)
            calculateSOQ(session,months,types,80,RUN_DATE,True, RUN_VERSION)
            calculateSOQ(session,months,types,95,RUN_DATE,False,RUN_VERSION)
            calculateSOQ(session,months,types,90,RUN_DATE,False,RUN_VERSION)
            calculateSOQ(session,months,types,85,RUN_DATE,False,RUN_VERSION)
            calculateSOQ(session,months,types,99,RUN_DATE,False,RUN_VERSION)
            calculateSOQ(session,months,types,80,RUN_DATE,False,RUN_VERSION)
    return session.table(SOQ_TABLE)
