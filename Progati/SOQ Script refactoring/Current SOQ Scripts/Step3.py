# The Snowpark package is required for Python Worksheets. 
# You can add more packages by selecting them using the Packages control and then importing them.
## Run Version 2 is with festive
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
from dateutil.relativedelta import relativedelta
import datetime
import pandas as pd
import numpy as np
import math

## April version 5 is marriage only view without any round/ceil
## April version 6 is no festive/marriage without any round/ceil
## Verion 7 is festive/marriage foreccast table with no round/ceil
#FORECAST_TABLE='MOP_DATABASE.SOQ.TEST_DATA_PARENT_DEALER_MODEL_FAMILY_CODE_MONTHLY_FORECASTS_WITH_PROP_SEPT_2024_VIEW_WITH_NEW_MODEL_FESTIVE_PREDICTION'
#FORECAST_TABLE='MOP_DATABASE.SOQ.TEST_DATA_PARENT_DEALER_MODEL_FAMILY_CODE_MONTHLY_FORECASTS_WITH_PROP_MAR_2025_VIEW_VERSION_FINAL'
#FORECAST_TABLE='MOP_DATABASE.SOQ.TEST_DATA_PARENT_DEALER_MODEL_FAMILY_CODE_MONTHLY_FORECASTS_WITH_PROP_APR_2025_WITH_HOLIDAYS_VIEW'


#FORECAST_TABLE='MOP_DATABASE.SOQ.TEST_DATA_PARENT_DEALER_MODEL_FAMILY_CODE_MONTHLY_FORECASTS_WITH_PROP_APR_2025_WO_HOLIDAYS_VIEW_ROUNDING'
#FORECAST_TABLE='MOP_DATABASE.SOQ.TEST_DATA_PARENT_DEALER_MODEL_FAMILY_CODE_MONTHLY_FORECASTS_WITH_PROP_APR_2025_WO_HOLIDAYS_VIEW_CEILING'

#FORECAST_TABLE='MOP_DATABASE.SOQ.TEST_DATA_PARENT_DEALER_MODEL_FAMILY_CODE_MONTHLY_FORECASTS_WITH_PROP_MAY_2025_WITH_FESTIVE_ONLY_MARRIAGE_VIEW'
#FORECAST_TABLE='MOP_DATABASE.SOQ.TEST_DATA_PARENT_DEALER_MODEL_FAMILY_CODE_MONTHLY_FORECASTS_WITH_PROP_MAY_2025_NORMAL_VIEW' 

#FORECAST_TABLE='MOP_DATABASE.SOQ.TEST_DATA_PARENT_DEALER_MODEL_FAMILY_CODE_MONTHLY_FORECASTS_WITH_PROP_APR_2025_WITH_HOLIDAYS_VIEW'
#FORECAST_TABLE= "MOP_DATABASE.SOQ.TEST_DATA_PARENT_DEALER_MODEL_FAMILY_CODE_MONTHLY_FORECASTS_WITH_PROP_MAY_2025_WITH_MARRIAGE_VIEW"
#FORECAST_TABLE='MOP_DATABASE.SOQ.TEST_DATA_PARENT_DEALER_MODEL_FAMILY_CODE_MONTHLY_FORECASTS_WITH_PROP_MAY_2025_WITH_MARRIAGE_FESTIVE_WITHOUT_MARCH_VIEW' #Use this for April End PRediction
#FORECAST_TABLE ='MOP_DATABASE.SOQ.TEST_DATA_PARENT_DEALER_MODEL_FAMILY_CODE_MONTHLY_FORECASTS_WITH_PROP_MAY_2025_WITH_MARRIAGE_FESTIVE_VIEW_WITHOUT_MAR'


#FORECAST_TABLE='MOP_DATABASE.SOQ.TEST_DATA_PARENT_DEALER_MODEL_FAMILY_CODE_MONTHLY_FORECASTS_WITH_PROP_MAY_2025_WITH_MARRIAGE_FESTIVE_VIEW' ## Use this for May First PRediction
PRED_LEVEL= "monthly" #"weekly"
#FORECAST_TABLE='MOP_DATABASE.SOQ.TEST_DATA_PARENT_DEALER_MODEL_FAMILY_CODE_WEEKLY_FORECASTS_JUN_2025_FINAL_FORECASTS' #used for June and May Weekly Forecasts
FORECAST_TABLE='MOP_DATABASE.SOQ.TEST_FORECASTS_WITH_MARKET_SHARE_SON_2025_VIEW_V2'


TEST_FORECAST_TABLE='MOP_DATABASE.SOQ.TEST_FORECASTS_PARENT_DEALER_MODEL_FAMILY_WITH_MARKET_SHARE_SON_2025'
TEST_DATA_TABLE='MOP_DATABASE.SOQ.TEST_DATA_MONTHLY_DEALER_MODEL_FAMILY_CODE_SON_2025_UPDATED'
RAW_TEST_TABLE=TEST_DATA_TABLE+"_WEEKLY_REGULARISED"
PREDICTION_TABLE='MOP_DATABASE.SOQ.SOQ_PREDICTION_FINAL_VERSION'

IS_OBD=True ## This will ensure we consider the stock, ecr of both present and previous SKU
OBD_MAPPING='MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW'

#CUSTOMER_TYPE_TO_CONSIDER=['Individual','Institutional']

CUSTOMER_TYPE_TO_CONSIDER=['Individual']

ECR_GROUP_BY=['PARENT_DEALER_CODE','UNIQUE FAMILY CODE','X_MONTH_NAME','MODEL'] ### GROUP BY FAMILY CODE

RUN_DATE=datetime.datetime.today().strftime('%Y%m%d')

RUN_VERSION=33

MID_DATE=15

BASE_SOQ_TABLE="MOP_DATABASE.SOQ.SOQ_BASE_TABLE_FINAL_CONCATENATED"

DEMAND_VARIABILITY_TABLE='MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_FINAL_VERSION'

DEMAND_VARIABILITY_FAMILY_TABLE='MOP_DATABASE.SOQ.DEMAND_VARIABILITY_SKU_MODEL_FAMILY_FINAL_VERSION'

TRANSIT_TABLE='MOP_DATABASE.SOQ.PARENT_DEALER_TRANSIT_TIME_SKU_NEW'

SKU_SUPERCEDENCE_MODEL_FAMILY='MOP_DATABASE.SOQ.SKU_SUPERCEDENCE_MODEL_FAMILY_MAY_2026_UPDATED_V2' ## Change to Mat

STOCK_DATE_TYPE=["first"] #,"mid"]    ## [2025-04-01: end ; 2025-05-01: first]
MONTHS=['2026-06-01'] #,'2024-09-01'] 

if IS_OBD:
    obd_flag='Y'
else: 
    obd_flag='N'
    
def fetchOBDData(session):
    obd_mapping=session.table(OBD_MAPPING).to_pandas()
    obd_mapping['SKU']=obd_mapping['PREVIOUS_OBD_SKU']
    return obd_mapping



def fetchStockData(session,date):

    '''
    In this function we consider the stock availability of the previous OBD variant SKU as well to get the Stock Availability if Is_OBD is true
    
    '''
    query=f'''
    SELECT DEALER_CODE,MODEL,SKU,CLOSING_STOCK as STK_AS_ON_DATE FROM ANALYTICS_DATABASE.ANALYTICS_SALES.STOCK_AVAILABILITY 
                WHERE CAL_DATE='{date}'
    '''
    stock_data=session.sql(query).to_pandas()
    if IS_OBD:
        obd_data=fetchOBDData(session)
    
        stock_data=pd.merge(stock_data,obd_data,on="SKU",how="left")
    
        ## Null in Current ObD SKU - means, current SKU is the same as sku
    
        stock_data['CURRENT_OBD_SKU'] = stock_data['CURRENT_OBD_SKU'].fillna(stock_data['SKU'])
    
        stk_data=stock_data.groupby(['CURRENT_OBD_SKU','DEALER_CODE','MODEL'])['STK_AS_ON_DATE'].sum().reset_index()
        
        stk_data=stk_data.rename(columns={'CURRENT_OBD_SKU':'SKU'})
        
        return stk_data
    return stock_data

def getParentDealerMapping(session):
    query='''
    SELECT DISTINCT X_DEALER_CODE_HIER AS DEALER_CODE,PAR_ORG_NAME FROM FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH WHERE X_DEALER_CODE_HIER IS NOT NULL
    '''
    parent_dealer_mapping=session.sql(query)
    parent_dealer_mapping=parent_dealer_mapping.to_pandas()
    parent_dealer_mapping['PARENT_DEALER_CODE']=parent_dealer_mapping['PAR_ORG_NAME'].apply(lambda x:str(x).split("-")[0].strip())
    return parent_dealer_mapping


def fetchSKUSupercedence(session):
    data= session.table(SKU_SUPERCEDENCE_MODEL_FAMILY).to_pandas()
    data=data.drop(['UPDATED_ON'],axis=1)
    print("Supercedence Shape ",data.shape)
    return data

    


def percentsku(row):
    if row['TOTAL_DEALER_ACTIVE_SKU_SALES']==0:
        #return 0
        return  1/row['NUM_ACTIVE_SKUS'] ## Equal proportion to all active SKU's
    else:
       
        
        if row['DEALER_SKU_SALES']==0:
            return 0
        else:
            dealer_sku_sales=row['DEALER_SKU_SALES']
        return dealer_sku_sales/row['TOTAL_DEALER_ACTIVE_SKU_SALES']
    

def getStockDate(date_period,planning_month):
    run_month=(datetime.datetime.strptime(planning_month, "%Y-%m-%d")- relativedelta(months=1))
    if date_period=="end":
        ## take last day of running month 
        return (datetime.datetime.strptime(planning_month, "%Y-%m-%d").replace(day=1) - relativedelta(days=1)).strftime("%Y-%m-%d")
    if date_period=="first":
        ## Take last day of previous month
        return (run_month.replace(day=1) -relativedelta(days=1)).strftime("%Y-%m-%d")
    if date_period=="mid":
        ### Take the max date in the month which is less than current date
        
        return run_month.replace(day=MID_DATE).strftime("%Y-%m-%d")

def write_to_snowflake(session,df,table_name, mode="append"):
    agg_data_sp=session.create_dataframe(df)
    agg_data_sp.write.mode(mode).save_as_table(table_name)

def processForecastTable(session,run_date,month,run_version):

    if  PRED_LEVEL=="weekly":
        test_df=session.table(TEST_FORECAST_TABLE).to_pandas()
        dates_df=session.table(RAW_TEST_TABLE).to_pandas()
        dates_df=dates_df[['Date','Regularized_Date','MONTH_DATE']].drop_duplicates()
        test_df=test_df.rename(columns={'TS':'Regularized_Date'})
        test_df=pd.merge(test_df,dates_df,on="Regularized_Date",how="left")
        test_df=test_df.groupby(['SERIES','MONTH_DATE'])['FORECAST'].sum().reset_index()
        test_df['PREDICTED_SALES']=test_df['FORECAST'].apply(lambda x: 0 if x<0 else round(x))

        test_df=test_df[['MONTH_DATE','SERIES','PREDICTED_SALES']]
        test_df.columns=['DATES','SERIES','PREDICTED_SALES']
        write_to_snowflake(session,test_df,FORECAST_TABLE,"overwrite")

        

    
    forecast_query=f'''
    SELECT * FROM {FORECAST_TABLE} WHERE DATES='{month}'
    '''
    df=session.sql(forecast_query).to_pandas()
    #df=session.table(FORECAST_TABLE).to_pandas()
    df['PARENT_DEALER_CODE']=df['SERIES'].apply(lambda x:x.split("_")[0].strip('"'))
    df['MODEL_FAMILY']=df['SERIES'].apply(lambda x:x.split("_")[1])
    df['FAMILY_CODE']=df['SERIES'].apply(lambda x:x.split("_")[2].strip('"'))

    df['UNIQUE FAMILY CODE']=df['MODEL_FAMILY']+"<>"+df['FAMILY_CODE']
    df['RUN_DATE']=run_date
    df['RUN_VERSION']=run_version
    session.create_dataframe(df).write.mode("append").save_as_table(PREDICTION_TABLE)

def fetchPredictionData(session,run_date,month,run_version):
    query=f'''
    SELECT * FROM {PREDICTION_TABLE} WHERE RUN_DATE='{run_date}' AND RUN_VERSION={run_version}
    '''
    return session.sql(query).to_pandas()

def getECR(session,customer_type_to_include,start_date,end_date):
    
    customer_type_to_include=["'"+types+"'" for types in customer_type_to_include ]
    customer_type_to_include=",".join(customer_type_to_include)

    

    query=f'''
            SELECT * FROM ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS WHERE X_CUSTOMER_TYPE IN ({customer_type_to_include}) 
            AND CAL_DATE>='{start_date}' AND CAL_DATE<'{end_date}'
      '''
    print(query)
    data=session.sql(query).to_pandas()
    data['DATE']=pd.to_datetime(data['CAL_DATE'])
    data['X_MONTH_NAME']=data['DATE'].apply(lambda x: x.strftime('%Y%m'))
    data['NET_SALES']=data['INVOICED_SALES']+data['CANCELLED_SALES']+data['RETURNED_SALES']
    return data

def ECRAggregation(session,run_date,sku_supercedence,parent_dealer_mapping):
    current_date=datetime.datetime.strptime(run_date, '%Y%m%d').replace(day=1)
    start_date=(current_date- relativedelta(months=3)).strftime("%Y-%m-%d")
    end_date=current_date.strftime("%Y-%m-%d")
    
    #- relativedelta(months=1))
    print(start_date)
    print(end_date)

    ecr_data=getECR(session,CUSTOMER_TYPE_TO_CONSIDER,start_date,end_date)
    print(ecr_data.shape)

    ## Map ECR to parent Dealer and SKU SUpercedence
    ecr_data=pd.merge(ecr_data,sku_supercedence,on=["SKU","MODEL"],how="left")

    ecr_data=pd.merge(ecr_data,parent_dealer_mapping[['DEALER_CODE','PARENT_DEALER_CODE']],on="DEALER_CODE",how="left")

    #ecr_data=ecr_data[~pd.isnull(ecr_data['SKUSTATUS'])]

    print(ecr_data.shape)

    if IS_OBD:
         ## ECR Data map with OBD
        obd_data=fetchOBDData(session)
        ecr_data=pd.merge(ecr_data,obd_data,on="SKU",how="left")

        ### Null in Current ObD SKU - means, current SKU is the same as sku
        ecr_data['CURRENT_OBD_SKU'] = ecr_data['CURRENT_OBD_SKU'].fillna(ecr_data['SKU'])
    
        ecr_data=ecr_data.rename(columns={'SKU':'ORIGINAL_SKU','CURRENT_OBD_SKU':'SKU'})
        
        
    return ecr_data



def getStockDataMapping(session,months,data,date_period,parent_dealer_mapping,sku_supercedence):
    stock_date=getStockDate(date_period,months)
    
    print(type(stock_date))
    ### Get the stock as on the given date
    
    stock_data=fetchStockData(session,stock_date)

    stock_data=pd.merge(stock_data,parent_dealer_mapping[['DEALER_CODE','PARENT_DEALER_CODE']],on="DEALER_CODE",how='left')

    stock_data=pd.merge(stock_data,sku_supercedence,on=["SKU","MODEL"],how="left") 

    stocks_sku=stock_data.groupby(['PARENT_DEALER_CODE','SKU'])['STK_AS_ON_DATE'].sum().reset_index()

    ## Get the active SKU's in each family

    #sku=sku_supercedence[~pd.isnull(sku_supercedence['SKUSTATUS'])]
    #active_families=sku['UNIQUE FAMILY CODE'],unique().tolist()
    #sku=sku[['SKU','UNIQUE FAMILY CODE']]
    #num_active=sku.groupby(['UNIQUE FAMILY CODE'])['SKU'].nunique().reset_index().rename(columns={'SKU':'count'}).sort_values(by="count",ascending=False)
    #active_families=num_active['UNIQUE FAMILY CODE'].tolist()
    num_active=sku_supercedence[sku_supercedence['SKUSTATUS']=='active'].groupby(['UNIQUE FAMILY CODE'])['SKU'].nunique().reset_index().rename(columns={'SKU':'count'}).sort_values(by="count",ascending=False)
    
    ### Filter out only for the active families
    #data=data[data['UNIQUE FAMILY CODE'].isin(active_families)]

    data=pd.merge(data,num_active,on="UNIQUE FAMILY CODE",how="left")
    data=data.rename(columns={'count':'NUM_ACTIVE_SKUS'})

    sku_data=sku_supercedence[sku_supercedence['SKUSTATUS']=='active']

    #inactive_families=sku_data[~sku_data['UNIQUE FAMILY CODE'].isin(active_families)]
    #session.create_dataframe(inactive_families).write.mode("overwrite").save_as_table('MOP_DATABASE.SOQ.INACTIVE_FAMILIES_TEMP')

    #sku_data=sku_data[sku_data['UNIQUE FAMILY CODE'].isin(active_families)]
    
    print("SKU DATA SHAPE")
    print(sku_data.shape)

    soq_data_sku=pd.merge(data,sku_data,on=["UNIQUE FAMILY CODE"],how="left")
    print(soq_data_sku.head())
    soq_data_sku['PARENT_DEALER_CODE']=soq_data_sku['PARENT_DEALER_CODE'].apply(lambda x:str(x))
    print("STOCK SKU")
    print(stocks_sku.shape)
    stocks_sku['PARENT_DEALER_CODE']=stocks_sku['PARENT_DEALER_CODE'].apply(lambda x:str(x))
    

    print("STOCKS DATA UNIQUE COMB")
    print(stocks_sku[['PARENT_DEALER_CODE','SKU']].drop_duplicates().nunique())


    soq_data_sku=pd.merge(soq_data_sku,stocks_sku,on=['PARENT_DEALER_CODE','SKU'],how="left")
    session.create_dataframe(soq_data_sku).write.mode("overwrite").save_as_table('MOP_DATABASE.SOQ.INACTIVE_FAMILIES_ACTIVE_FAMILIES_MERGE_TEMP')

    soq_data_sku=soq_data_sku[~pd.isnull(soq_data_sku['NUM_ACTIVE_SKUS'])]
    
    return soq_data_sku

def calculateDemandVariability(group):
    #dealer_code=group.name[0]
    #family_code=group.name[1]
    product_sales=group['NET_SALES']
    if len(product_sales)>2:
        errors = product_sales.diff().dropna()
        demand_variability = errors.std()
        return demand_variability
    else:
        return 1


def createDemandVariability(session,months,run_date=RUN_DATE,run_version=RUN_VERSION):
    ECR_GROUP_BY=['PARENT_DEALER_CODE','UNIQUE FAMILY CODE','X_MONTH_NAME','SKU']
    start_date=(datetime.datetime.strptime(run_date, "%Y%m%d").replace(day=1)- relativedelta(months=12)).strftime("%Y-%m-%d")
    end_date=(datetime.datetime.strptime(run_date, "%Y%m%d").replace(day=1)).strftime("%Y-%m-%d")
    print(start_date,end_date)
    ecr_data=getECR(session,CUSTOMER_TYPE_TO_CONSIDER,start_date,end_date)
    ## Map ECR to parent Dealer and SKU SUpercedence

    parent_dealer_mapping=getParentDealerMapping(session)
    sku_supercedence=fetchSKUSupercedence(session)

    ecr_data=pd.merge(ecr_data,sku_supercedence,on=["SKU","MODEL"],how="left")

    ecr_data=pd.merge(ecr_data,parent_dealer_mapping[['DEALER_CODE','PARENT_DEALER_CODE']],on="DEALER_CODE",how="left")

    if IS_OBD:
        ## Merge ecr data data with OBD data rename CURRENT_OBD_VERSION as SKU and go ahead
        obd_data=fetchOBDData(session)
        ecr_data=pd.merge(ecr_data,obd_data,on="SKU",how="left")

        ### Null in Current ObD SKU - means, current SKU is the same as sku
        ecr_data['CURRENT_OBD_SKU'] = ecr_data['CURRENT_OBD_SKU'].fillna(ecr_data['SKU'])

        ecr_data=ecr_data.rename(columns={'SKU':'ORIGINAL_SKU','CURRENT_OBD_SKU':'SKU'})
    
    total_sales_by_month=ecr_data.groupby(ECR_GROUP_BY)['NET_SALES'].sum().reset_index()
    total_sales_by_month=total_sales_by_month.sort_values(by=['PARENT_DEALER_CODE','SKU','X_MONTH_NAME'])
    total_sales_by_month['NET_SALES']=total_sales_by_month['NET_SALES'].apply(lambda x:float(x))
    ### For each Product, for Each Dealer let us calculate the demand variability
    demand_variability=total_sales_by_month.groupby(['PARENT_DEALER_CODE','UNIQUE FAMILY CODE','SKU']).apply(calculateDemandVariability).reset_index().rename(columns={0:"DEMAND_VARIABILITY"})
    demand_variability['PLANNING_MONTH']=months 
    demand_variability['ECR_START_DATE']=start_date
    demand_variability['ECR_END_DATE']=end_date
    demand_variability['RUN_DATE']=run_date
    demand_variability['RUN_VERSION']=run_version
    demand_variability['IS_OBD']=obd_flag
    agg_data_sp=session.create_dataframe(demand_variability)
    agg_data_sp.write.mode("append").save_as_table(DEMAND_VARIABILITY_TABLE)
    

def createDemandVariabilityByFamily(session,months,run_date=RUN_DATE,run_version=RUN_VERSION):
    ECR_GROUP_BY=['PARENT_DEALER_CODE','UNIQUE FAMILY CODE','X_MONTH_NAME']
    start_date=(datetime.datetime.strptime(run_date, "%Y%m%d").replace(day=1)- relativedelta(months=12)).strftime("%Y-%m-%d")
    end_date=(datetime.datetime.strptime(run_date, "%Y%m%d").replace(day=1)).strftime("%Y-%m-%d")
    print(start_date,end_date)
    ecr_data=getECR(session,CUSTOMER_TYPE_TO_CONSIDER,start_date,end_date)
    ## Map ECR to parent Dealer and SKU SUpercedence

    parent_dealer_mapping=getParentDealerMapping(session)
    sku_supercedence=fetchSKUSupercedence(session)

    ecr_data=pd.merge(ecr_data,sku_supercedence,on=["SKU","MODEL"],how="left")

    ecr_data=pd.merge(ecr_data,parent_dealer_mapping[['DEALER_CODE','PARENT_DEALER_CODE']],on="DEALER_CODE",how="left")
    if IS_OBD:
        ## Merge ecr data data with OBD data rename CURRENT_OBD_VERSION as SKU and go ahead
        obd_data=fetchOBDData(session)
        ecr_data=pd.merge(ecr_data,obd_data,on="SKU",how="left")

        ### Null in Current ObD SKU - means, current SKU is the same as sku
        ecr_data['CURRENT_OBD_SKU'] = ecr_data['CURRENT_OBD_SKU'].fillna(ecr_data['SKU'])

        ecr_data=ecr_data.rename(columns={'SKU':'ORIGINAL_SKU','CURRENT_OBD_SKU':'SKU'})
    total_sales_by_month=ecr_data.groupby(ECR_GROUP_BY)['NET_SALES'].sum().reset_index()
    total_sales_by_month=total_sales_by_month.sort_values(by=['PARENT_DEALER_CODE','UNIQUE FAMILY CODE','X_MONTH_NAME'])
    total_sales_by_month['NET_SALES']=total_sales_by_month['NET_SALES'].apply(lambda x:float(x))
    ### For each Product, for Each Dealer let us calculate the demand variability
    demand_variability=total_sales_by_month.groupby(['PARENT_DEALER_CODE','UNIQUE FAMILY CODE']).apply(calculateDemandVariability).reset_index().rename(columns={0:"DEMAND_VARIABILITY"})
    demand_variability['PLANNING_MONTH']=months 
    demand_variability['ECR_START_DATE']=start_date
    demand_variability['ECR_END_DATE']=end_date
    demand_variability['RUN_DATE']=run_date
    demand_variability['RUN_VERSION']=run_version
    demand_variability['IS_OBD']=obd_flag
    agg_data_sp=session.create_dataframe(demand_variability)
    agg_data_sp.write.mode("append").save_as_table(DEMAND_VARIABILITY_FAMILY_TABLE)
    
    
def main(session: snowpark.Session): 
    # Your code goes here, inside the "main" handler
    parent_dealer_mapping=getParentDealerMapping(session)
    sku_supercedence=fetchSKUSupercedence(session) # UNIQUE FAMILY CODE -> MODEL_FAMILY<>DRUM<>
    
    for months in MONTHS:
    
        processForecastTable(session, RUN_DATE,months,RUN_VERSION)

        prediction_query=f'''SELECT DISTINCT * FROM {PREDICTION_TABLE} WHERE RUN_DATE='{RUN_DATE}' AND DATES='{months}' AND RUN_VERSION={RUN_VERSION}'''
    
        data=session.sql(prediction_query).to_pandas()
        data['PARENT_DEALER_CODE']=data["PARENT_DEALER_CODE"].apply(lambda x:str(x))
        data['DATES_STR']=data['DATES'].apply(lambda x: x.strftime("%Y-%m-%d"))
        print(data.shape)
        
        for stock_period in STOCK_DATE_TYPE:
            soq_data_sku=getStockDataMapping(session,months,data,stock_period,parent_dealer_mapping,sku_supercedence)
            ecr_data=ECRAggregation(session,RUN_DATE,sku_supercedence,parent_dealer_mapping)
            
            ## Sales of the Dealer at FAMILY LEVEL IN LAST THREE MONTHS
            dealer_family_sales=ecr_data.groupby(['PARENT_DEALER_CODE','UNIQUE FAMILY CODE'])['NET_SALES'].sum().reset_index().rename(columns={'NET_SALES':'DEALER_FAMILY_CODE_NET_SALES'})
    
            dealer_family_sku_sales=ecr_data.groupby(['PARENT_DEALER_CODE','UNIQUE FAMILY CODE','SKU'])['NET_SALES'].sum().reset_index().rename(columns={'NET_SALES':'DEALER_SKU_SALES'})
    
            soq_data_sku['PARENT_DEALER_CODE']=soq_data_sku['PARENT_DEALER_CODE'].apply(lambda x:str(int(x)))
    
            soq_data_sku=pd.merge(soq_data_sku,dealer_family_sales,on=['PARENT_DEALER_CODE','UNIQUE FAMILY CODE'],how="left")
    
            soq_data_sku.loc[pd.isnull(soq_data_sku['DEALER_FAMILY_CODE_NET_SALES']),'DEALER_FAMILY_CODE_NET_SALES']=0
            
            soq_data_sku=pd.merge(soq_data_sku,dealer_family_sku_sales,on=['PARENT_DEALER_CODE','UNIQUE FAMILY CODE','SKU'],how="left")

            active_family_dealer_sku_sales=soq_data_sku.groupby(['PARENT_DEALER_CODE','UNIQUE FAMILY CODE'])['DEALER_SKU_SALES'].sum().reset_index().rename(columns={'DEALER_SKU_SALES':'TOTAL_DEALER_ACTIVE_SKU_SALES'})

            soq_data_sku=pd.merge(soq_data_sku,active_family_dealer_sku_sales,on=['PARENT_DEALER_CODE','UNIQUE FAMILY CODE'],how="left")
            soq_data_sku.loc[pd.isnull(soq_data_sku['DEALER_SKU_SALES']),'DEALER_SKU_SALES']=0

            ## Added this line to meet OBD constraint - where the family sales may be there but that SKU sale may be zero
            dealer_sku_total=soq_data_sku.groupby(['PARENT_DEALER_CODE','SKU_UNIQUE_FAMILY_CODE'])['DEALER_SKU_SALES'].sum().reset_index().rename(columns={'DEALER_SKU_SALES':'DEALER_ACTIVE_SKU_TOTAL_SALES'})
            soq_data_sku=pd.merge(soq_data_sku,dealer_sku_total,on=['PARENT_DEALER_CODE','SKU_UNIQUE_FAMILY_CODE'],how="left")
            
            ### End of chnage to meet OBD
            
            
            soq_data_sku['PERCENT_PROPORTION']=soq_data_sku.apply(lambda row:round(percentsku(row),5),axis=1)
            #soq_data_sku['PREDICTED_SALES_SKU'] = soq_data_sku.apply(lambda row: row['PREDICTED_SALES'] if row['NUM_ACTIVE_SKUS'] == 1  else round(row['PERCENT_PROPORTION'] * row['PREDICTED_SALES'],0), axis=1)    
            soq_data_sku['PREDICTED_SALES_SKU'] = soq_data_sku.apply(lambda row: row['PREDICTED_SALES'] if row['NUM_ACTIVE_SKUS'] == 1  else row['PERCENT_PROPORTION'] * row['PREDICTED_SALES'], axis=1)    
            
            transit_data=session.table(TRANSIT_TABLE).to_pandas()
            soq_data_sku.drop(['DEALER_ACTIVE_SKU_TOTAL_SALES','TOTAL_DEALER_ACTIVE_SKU_SALES'],axis=1,inplace=True)
            print(soq_data_sku.shape)
            soq_final_data=pd.merge(soq_data_sku,transit_data,on=['PARENT_DEALER_CODE','SKU'],how="left")
            print(soq_final_data.shape)
            stock_date=getStockDate(stock_period,months)
            soq_final_data['STOCK_DATE_PERIOD']=stock_period
            soq_final_data['STOCK_DATE']=stock_date
            soq_final_data['PLANNING_MONTH']=months
            soq_final_data['RUN_DATE']=RUN_DATE
            
            soq_final_data.drop_duplicates(inplace=True)

            soq_final_data['RUN_VERSION']=RUN_VERSION
            soq_final_data['IS_OBD']=obd_flag
            agg_data_sp=session.create_dataframe(soq_final_data)
            agg_data_sp.write.mode("append").save_as_table(BASE_SOQ_TABLE)
    
    for months in MONTHS:
        print(months)
        createDemandVariability(session,months,RUN_DATE,RUN_VERSION)
        createDemandVariabilityByFamily(session,months,RUN_DATE, RUN_VERSION)

    return session.table(DEMAND_VARIABILITY_FAMILY_TABLE)
        
    
