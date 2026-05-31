
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import pandas as pd
import re
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

transit_table_name="SAP_DATA_DATABASE.SAP_DATA.ZSDTTPTACTFRT"

VSTEL_MAPPING={"HHDS":"HHHD","HHGS":"HHHG","HHUS":"HHHU","HS4N":"HM4N","HS5V":"HM5V","HS6C":"HM6C"}

#CUSTOMER_TYPE_TO_CONSIDER=['Individual','Institutional']

CUSTOMER_TYPE_TO_CONSIDER=['Individual']   # ADD Institutional if needed 

ECR_GROUP_BY=['PARENT_DEALER_CODE','UNIQUE FAMILY CODE','X_MONTH_NAME','MODEL']


START_DATE='2023-01-01'

TRAIN_END_DATE='2026-05-01' ### '2025-04-01'

MAX_DATE="2026-08-01" ## 2025-07-01 


AGG_TYPE="monthly"    #"weekly"
#AGG_TYPE="weekly"  

TRAIN_DATA_TABLE="MOP_DATABASE.SOQ.TRAIN_DATA_MONTHLY_DEALER_MODEL_FAMILY_CODE_MAY_2026_UPDATED_V2" ### Make it May wherever Apr

TEST_DATA_TABLE="MOP_DATABASE.SOQ.TEST_DATA_MONTHLY_DEALER_MODEL_FAMILY_CODE_MAY_2026_UPDATED_V2"

ECR_DAILY_TABLE="MOP_DATABASE.SOQ.ECR_DAILY_MERGED_SKU"

ECR_DATA_TABLE='MOP_DATABASE.SOQ.ECR_DEALER_SKU_SALES_MAY_2026_V2'

SKU_SUPERCEDENCE_MODEL_FAMILY='MOP_DATABASE.SOQ.SKU_SUPERCEDENCE_MODEL_FAMILY_MAY_2026_UPDATED_V2'

FESTIVE_TABLES='MOP_DATABASE.SOQ.FESTIVE_DAYS_SOQ'

#WEEKLY_FESTIVE_DATA='MOP_DATABASE.SOQ.FESTIVE_DAYS_WITH_MARRIAGE_WEEK_WISE'
WEEKLY_FESTIVE_DATA='MOP_DATABASE.SOQ.FESTIVE_MARRIAGE_DATA_WEEK_WISE'

def write_to_snowflake(session,df,table_name, mode="append"):
    agg_data_sp=session.create_dataframe(df)
    agg_data_sp.write.mode(mode).save_as_table(table_name)



def first_dates_of_months(start_date, end_date):
    # Generate a range of dates from start_date to end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Extract the first date of each month
    first_dates = date_range[date_range.is_month_start]
    first_dates = [dates.to_pydatetime() for dates in first_dates]
    return first_dates

def getDailyECR(session,customer_type_to_include,start_date):
    customer_type_to_include=["'"+types+"'" for types in customer_type_to_include ]
    customer_type_to_include=",".join(customer_type_to_include)

   
    

    query=f'''
            SELECT DEALER_CODE,MODEL,SKU,CAL_DATE,MONTH(CAL_DATE) AS CAL_MONTH,YEAR(CAL_DATE) AS CAL_YEAR,SUM(INVOICED_SALES) AS INVOICED_SALES,SUM(CANCELLED_SALES) AS CANCELLED_SALES,SUM(RETURNED_SALES) AS RETURNED_SALES FROM ANALYTICS_DATABASE.ANALYTICS_SALES.CUSTOMER_RETAILS  
            WHERE X_CUSTOMER_TYPE IN ({customer_type_to_include}) 
            AND CAL_DATE>='{start_date}' 
            GROUP BY  DEALER_CODE,MODEL,SKU, CAL_DATE
      '''
    ecr_sales=session.sql(query).to_pandas()
    ecr_sales['NET_SALES']=ecr_sales['INVOICED_SALES']+ecr_sales['CANCELLED_SALES']+ecr_sales['RETURNED_SALES']
    return ecr_sales


def fetchTransitData(session:snowpark.Session):
    transit_table=session.table(transit_table_name)

    transit_data=transit_table.to_pandas()

    transit_data=transit_data[['KUNNR','VSTEL','DATB','LDAYS']]

    transit_data=transit_data[transit_data['DATB']!='00000000']


    transit_data['PLANT']=transit_data['VSTEL'].apply(lambda x:VSTEL_MAPPING[x] if x in VSTEL_MAPPING else "")
    
    transit_data=transit_data[~pd.isnull(transit_data['KUNNR'])]

    transit_data['Customer']=transit_data['KUNNR'].apply(lambda x:x.lstrip("0"))


    max_date=transit_data.groupby(['Customer','VSTEL','PLANT'])['DATB'].max().reset_index().rename(columns={"DATB":'MAX_DATE'})

    transit_data=pd.merge(transit_data,max_date,on=["Customer","PLANT","VSTEL"],how="left")

    transit_data=transit_data[transit_data['MAX_DATE']==transit_data['DATB']]
    
    transit_data=transit_data[['Customer','LDAYS',"PLANT",'VSTEL']]
    
    transit_data.columns=['DEALER_CODE','TRANSIT_TIME','PLANT','VSTEL']
    
    transit_data['TRANSIT_TIME']=transit_data['TRANSIT_TIME'].apply(lambda x:int(x))

    return transit_data

def fetchParentDealerMapping(session:snowpark.Session):
    query='''
    SELECT DISTINCT X_DEALER_CODE_HIER AS DEALER_CODE,PAR_ORG_NAME FROM FIVETRAN_DATABASE.ORACLE_LDP_OLAP_SCHEMA.WC_INT_ORG_DH WHERE X_DEALER_CODE_HIER IS NOT NULL
    '''
    parent_dealer_mapping=session.sql(query)
    parent_dealer_mapping=parent_dealer_mapping.to_pandas()
    parent_dealer_mapping['PARENT_DEALER_CODE']=parent_dealer_mapping['PAR_ORG_NAME'].apply(lambda x:str(x).split("-")[0].strip())

    return parent_dealer_mapping

def fetchSkuPlantMapping(session:snowpark.Session):
    sku_plant_mapping_query='''
            SELECT * FROM MOP_DATABASE.SOQ.SKU_PLANT_MAPPING_VIEW
    '''
    sku_plant_mapping=session.sql(sku_plant_mapping_query)

    sku_plant_mapping=sku_plant_mapping.to_pandas()

    return sku_plant_mapping
    
def fetchAggregateTransitTime(session:snowpark.Session):
    query='''
    SELECT MAX(TRANSIT_TIME) as MAX_LEAD_TIME, MIN(TRANSIT_TIME) as MIN_LEAD_TIME,
    AVG(TRANSIT_TIME) as AVG_LEAD_TIME, SKU, PARENT_DEALER_CODE FROM MOP_DATABASE.SOQ.TRANSIT_DEALER_SKU_PLANT_MAPPING_NEW WHERE PRODUCTION=TRUE
    GROUP BY  SKU, PARENT_DEALER_CODE
    '''
    df=session.sql(query)
    
    return df



def replace_before_first_chevron(row):
    parts = row['UNIQUEFAMILYCODE'].split('<>', 1)  # Split at the first occurrence of '<>'
    return f"{row['MODEL_FAMILY']}<>{parts[1]}"


def fetchOBDData(session):

    query=f'''
    SELECT * FROM MOP_DATABASE.SOQ.OBD2_MAPPING_VIEW
    '''
    df=session.sql(query).to_pandas()
    df['SKU'] = df['PREVIOUS_OBD_SKU']
    
    return df

def fetchSKUSupercedence(session):
    query=f'''
    SELECT * FROM MOP_DATABASE.SOQ.SKU_SUPERCEDENCE
    '''
    data=session.sql(query).to_pandas()
    #data=data.rename(columns={"UNIQUEFAMILYCODE":'UNIQUE FAMILY CODE'})

    query_1 = f'''
    SELECT * FROM MOP_DATABASE.SOQ.MODEL_FAMILY_MAPPING
    '''
    data_1 = session.sql(query_1).to_pandas()
    
    result = pd.merge(data, data_1, on='MODEL', how='left')
    result['SKU_UNIQUE_FAMILY_CODE']=result['UNIQUEFAMILYCODE'].copy()



    result['UNIQUEFAMILYCODE'] = result.apply(replace_before_first_chevron, axis=1)
    
    result=result.rename(columns={"UNIQUEFAMILYCODE":'UNIQUE FAMILY CODE'})

    

    session.create_dataframe(result).write.mode("overwrite").save_as_table(SKU_SUPERCEDENCE_MODEL_FAMILY)

    return result


def get_week_start(date):
    day = date.day
    if 1 <= day <= 7:
        week_start_day = 1
    elif 8 <= day <= 14:
        week_start_day = 8
    elif 15 <= day <= 21:
        week_start_day = 15
    else:
        week_start_day = 22
    return pd.Timestamp(date.year, date.month, week_start_day)




    
def ECRDealerSalesModelFamily(session,agg_period,customer_type,start_date):
    ecr_data=getDailyECR(session,customer_type,start_date)

    sku_supercedence=fetchSKUSupercedence(session)

    obd_data= fetchOBDData(session)
    ecr_sales=pd.merge(ecr_data,obd_data,on="SKU",how="left")
   
        ## Null in Current ObD SKU - means, current SKU is the same as sku
   
    ecr_sales['CURRENT_OBD_SKU'] = ecr_sales['CURRENT_OBD_SKU'].fillna(ecr_sales['SKU'])
   
    ecr_sales=ecr_sales.drop(['SKU'],axis=1)
    ecr_sales=ecr_sales.rename(columns={'CURRENT_OBD_SKU':'SKU'})
    
    sku_supercedence['FAMILY_CODE']=sku_supercedence.apply(lambda row:row['UNIQUE FAMILY CODE'].replace(str(row['MODEL_FAMILY']),"").strip("<").strip(">"),axis=1)
    parent_dealer_mapping=fetchParentDealerMapping(session)
    
    ecr_sales=pd.merge(ecr_sales,parent_dealer_mapping[['DEALER_CODE','PARENT_DEALER_CODE']],on="DEALER_CODE",how="left") 
 
    ecr_sales=pd.merge(ecr_sales,sku_supercedence, on=['MODEL','SKU'],how="inner")

    

    session.create_dataframe(ecr_sales).write.mode("overwrite").save_as_table(ECR_DAILY_TABLE)
    if agg_period=="monthly":
       
        print("Aggregating")
        ecr_dealer_sales=ecr_sales.groupby(['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE','CAL_MONTH','CAL_YEAR'])['NET_SALES'].sum().reset_index()
            
        
        ecr_dealer_sales['Date']=ecr_dealer_sales.apply(lambda row:datetime.datetime(row['CAL_YEAR'],row['CAL_MONTH'],1),axis=1)
    elif agg_period=="weekly":
        print("aggregating Weekly")
        ecr_sales['Date']=ecr_sales['CAL_DATE'].apply(lambda x:get_week_start(x))
        ecr_dealer_sales=ecr_sales.groupby(['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE','CAL_MONTH','CAL_YEAR','Date'])['NET_SALES'].sum().reset_index()
        
    else:
       
        ecr_dealer_sales=ecr_sales.groupby(['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE','CAL_DATE','CAL_MONTH','CAL_YEAR'])['NET_SALES'].sum().reset_index()
        
        ecr_dealer_sales['Date']=pd.to_datetime(ecr_dealer_sales['CAL_DATE'])
        
    return ecr_dealer_sales

def getValidCombinations(session,ecr_dealer_sales, start_date):
    valid_df=ecr_dealer_sales[ecr_dealer_sales['Date']>=start_date]
    valid_df_count=valid_df. groupby(['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE'])['NET_SALES'].sum().reset_index()
    valid_df_count=valid_df_count.sort_values(by="NET_SALES",ascending=False)
    


    unique_combinations=valid_df_count[['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE']].drop_duplicates()
    unique_combinations['is_active_group']=1
    ecr_dealer_sales=pd.merge(ecr_dealer_sales,unique_combinations,on=['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE'],how="left")
    dropped_sales=ecr_dealer_sales[pd.isnull(ecr_dealer_sales['is_active_group'])]
    session.create_dataframe(dropped_sales).write.mode("overwrite").save_as_table("MOP_DATABASE.SOQ.DROPPED_COMBINATIONS_ECR_TEMP_DEC")
    
    ecr_dealer_sales=ecr_dealer_sales[~pd.isnull(ecr_dealer_sales['is_active_group'])]
    return ecr_dealer_sales


def create_test_dates(session,sales_data,max_end_period,train_end=TRAIN_END_DATE):
    grouped = sales_data.groupby(['PARENT_DEALER_CODE', 'MODEL_FAMILY','FAMILY_CODE'])
    results = []
    #print(len(grouped))
    for (parent_dealer_code, model_family,family_code), group in grouped:

        
        # Find the minimum date in the group
        start_date = '2025-01-01'
        
        # Generate all relevant dates
        dates = first_dates_of_months(start_date, max_end_period)
        
        # Create a DataFrame for this combination
        
        temp_df = pd.DataFrame({
            'Date': dates,
            'PARENT_DEALER_CODE': parent_dealer_code,
            'MODEL_FAMILY': model_family,
            'FAMILY_CODE':family_code
        })
        results.append(temp_df)
        
        
    results_df=pd.concat(results)   
    if results_df.shape[0]>0:
        festive_dates_month=session.table("WORK_DATABASE.MOP.FESTIVE_INDIAN_SEASON_AGG_MONTH").to_pandas()
        festive_dates_month=festive_dates_month.rename(columns={'MONTH_DATE':'Date'})
        temp_df=pd.merge(results_df,festive_dates_month,on="Date",how="left")
        final_data=pd.merge(temp_df,sales_data[['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE','Date','NET_SALES']],on=['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE','Date'],how="left")
        final_data.loc[pd.isnull(final_data['NET_SALES']),'NET_SALES']=0
        final_data['PARENT_DEALER_CODE_MODEL_FAMILY']=final_data.apply(lambda row:row['PARENT_DEALER_CODE']+"_"+row['MODEL_FAMILY']+"_"+row['FAMILY_CODE'],axis=1)
        final_data=final_data.fillna(0)
        #train_df=final_data[final_data['Date']<train_end]
        test_df=final_data[final_data['Date']>=train_end]
        #write_to_snowflake(session,train_df,TRAIN_DATA_TABLE,"append")
        write_to_snowflake(session,test_df,TEST_DATA_TABLE,"append")
 

import pandas as pd
from typing import List, Tuple

# Updated helper function
def custom_weekly_dates_with_next(start_date_str: str, end_date_str: str) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    # Convert strings to Timestamps
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    
    dates = []
    current_date = pd.Timestamp(start_date.year, start_date.month, 1)
    
    while current_date <= end_date:
        for week_start_day in [1, 8, 15, 22]:
            week_date = pd.Timestamp(current_date.year, current_date.month, week_start_day)
            if week_date >= start_date and week_date <= end_date:
                dates.append(week_date)
        # Move to the first of next month
        if current_date.month == 12:
            current_date = pd.Timestamp(current_date.year + 1, 1, 1)
        else:
            current_date = pd.Timestamp(current_date.year, current_date.month + 1, 1)
    
    # Compute Date and Next_Week_Date with datetime.datetime
    date_pairs = []
    for i in range(len(dates)):
        current_week = dates[i].to_pydatetime()
        if i + 1 < len(dates):
            next_week = dates[i + 1].to_pydatetime()
        else:
            # Compute next week start date manually
            if current_week.day == 1:
                next_week = pd.Timestamp(current_week.year, current_week.month, 8)
            elif current_week.day == 8:
                next_week = pd.Timestamp(current_week.year, current_week.month, 15)
            elif current_week.day == 15:
                next_week = pd.Timestamp(current_week.year, current_week.month, 22)
            elif current_week.day == 22:
                if current_week.month == 12:
                    next_week = pd.Timestamp(current_week.year + 1, 1, 1)
                else:
                    next_week = pd.Timestamp(current_week.year, current_week.month + 1, 1)
            next_week = next_week.to_pydatetime()
        date_pairs.append((current_week, next_week))
    
    return date_pairs


# Updated create_dates function with Next_Week_Date continuing logic
def create_dates_weekly(session, sales_data, max_end_period_str, train_end=TRAIN_END_DATE):
    grouped = sales_data.groupby(['PARENT_DEALER_CODE', 'MODEL_FAMILY', 'FAMILY_CODE'])
    results = []
    

# Ensure sales_data['Date'] is also datetime64[ns]
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    for (parent_dealer_code, model_family, family_code), group in grouped:
        # Convert min date in the group to string, then to Timestamp
        start_date_str = min(group['Date']).strftime('%Y-%m-%d')
        
        # Generate custom weekly start date pairs
        date_pairs = custom_weekly_dates_with_next(start_date_str, max_end_period_str)
        
        # Create a DataFrame for this combination
        temp_df = pd.DataFrame({
            'Date': [d[0] for d in date_pairs],
            'Next_Week_Date': [d[1] for d in date_pairs],
            'PARENT_DEALER_CODE': parent_dealer_code,
            'MODEL_FAMILY': model_family,
            'FAMILY_CODE': family_code
        })
        results.append(temp_df)
    
    results_df=pd.concat(results).reset_index(drop=True)
    results_df['Date'] = pd.to_datetime(results_df['Date'])
    results_df['Next_Week_Date'] = pd.to_datetime(results_df['Next_Week_Date'])
    if results_df.shape[0]>0:
        ## First merge with the sales data
        final_data=pd.merge(results_df,sales_data[['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE','Date','NET_SALES']],on=['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE','Date'],how="left")
        final_data.loc[pd.isnull(final_data['NET_SALES']),'NET_SALES']=0
        final_data['PARENT_DEALER_CODE_MODEL_FAMILY']=final_data.apply(lambda row:row['PARENT_DEALER_CODE']+"_"+row['MODEL_FAMILY']+"_"+row['FAMILY_CODE'],axis=1)
        ## Now merge with Week Wise Festive & Marriage dates on Date
        festive_data=session.table(WEEKLY_FESTIVE_DATA).to_pandas()
        
        festive_data=festive_data.rename(columns={'WEEK_LABEL':'Date'})
        festive_data['Date'] = pd.to_datetime(festive_data['Date'])
        final_data=pd.merge(final_data,festive_data,on="Date",how="left")
        
        ### Merge with only Marriage Days for Next Week
        marriage_data=festive_data[['Date','MARRIAGE_DAY']]
        marriage_data=marriage_data.rename(columns={'Date':'Next_Week_Date','MARRIAGE_DAY':'NEXT_WEEK_MARRIAGE_DAYS'})
        final_data=pd.merge(final_data,marriage_data,on='Next_Week_Date')
        
        final_data=final_data.fillna(0) 
        train_end_ts = pd.to_datetime(train_end)
        final_data['Data']=final_data['Date'].apply(lambda x:"train" if x<train_end_ts else "test")
        #final_data['Data']=final_data['Date'].apply(lambda x:"train" if x<train_end else "test")
        final_data['MONTH'] = final_data['Date'].dt.month
        final_data['YEAR']=final_data['Date'].dt.year
        marriage_df=session.table("MOP_DATABASE.SOQ.FESTIVE_DAYS_WITH_MARRIAGE").to_pandas()
        
        final_data['WEEK']=final_data['Date'].apply(lambda x:get_week_label(x))
        final_data['NUMBER_OF_DAYS_WEEK']=final_data['Date'].apply(lambda x:get_number_of_days_in_week(x))
    
        final_data['NUMBER_OF_DAYS_NEXT_WEEK']=final_data['Next_Week_Date'].apply(lambda x: get_number_of_days_in_week(x))
        marriage_df['MONTH_DATE'] = marriage_df['DATE'].values.astype('datetime64[M]')
        month_marriage_days = marriage_df.groupby('MONTH_DATE')['MARRIAGE_DAY'].sum().reset_index()
        month_marriage_days.rename(columns={'MARRIAGE_DAY': 'TOTAL_MARRIAGE_DAYS_IN_MONTH'}, inplace=True)
        
        final_data['MONTH_DATE']=final_data['Date'].values.astype('datetime64[M]')
        final_data=pd.merge(final_data,month_marriage_days,on="MONTH_DATE",how='left')
        final_data['PROP_MARRIAGE_DAYS']=final_data['MARRIAGE_DAY']/final_data['NUMBER_OF_DAYS_WEEK']
        final_data['PROP_NEXT_WEEK_MARRIAGE_DAYS']=final_data['NEXT_WEEK_MARRIAGE_DAYS']/final_data['NUMBER_OF_DAYS_NEXT_WEEK']
        final_data['PERCENT_MARRIAGE_DAYS_CURRENT_WEEK']=final_data.apply(lambda row: get_percent_marriagedays(row['MARRIAGE_DAY'],row['TOTAL_MARRIAGE_DAYS_IN_MONTH']),axis=1)
    
        #combined_labels = ['Low', 'Medium', 'High']
        #bin_edges,adjusted_labels=create_bins("PERCENT_MARRIAGE_DAYS_CURRENT_WEEK",final_data,combined_labels)
        
        # final_data['MARRIAGE_WEEK_BIN'] = pd.cut(
        # final_data['PERCENT_MARRIAGE_DAYS_CURRENT_WEEK'],
        # bins=bin_edges,
        # labels=adjusted_labels,
        # include_lowest=True
        #  )
        # final_data['MARRIAGE_WEEK_BIN']=final_data.apply(lambda row:"No" if row['PERCENT_MARRIAGE_DAYS_CURRENT_WEEK']==0 else row['MARRIAGE_WEEK_BIN'],axis=1)


        # ### Is peak marriage season - Y/N
        # combined_labels=['Low','Medium','High']
        # bin_edges,adjusted_labels=create_bins("TOTAL_MARRIAGE_DAYS_IN_MONTH",final_data,combined_labels)
        # #combined_df['MARRIAGE_MONTH_BIN']=combined_df['TOTAL_MARRIAGE_DAYS_IN_MONTH'].apply(lambda x:assign_marriage_bin(x,bin_edges,adjusted_labels))
    
        
        
        # final_data['MARRIAGE_MONTH_BIN']=pd.cut(
        #     final_data['TOTAL_MARRIAGE_DAYS_IN_MONTH'],
        #     bins=bin_edges,
        #     labels=adjusted_labels,
        #     include_lowest=True
        # )
    
        # final_data['MARRIAGE_MONTH_BIN']=final_data.apply(lambda row:"No" if row['TOTAL_MARRIAGE_DAYS_IN_MONTH']==0 else row['MARRIAGE_MONTH_BIN'],axis=1)
                
        train_df=final_data[final_data['Date']<train_end]
        test_df=final_data[final_data['Date']>=train_end]

        if train_df.shape[0]>0:
            write_to_snowflake(session,train_df,TRAIN_DATA_TABLE+"_WEEKLY","append")
            write_to_snowflake(session,test_df,TEST_DATA_TABLE+"_WEEKLY","append")



def create_regularised_dates(session):
    train_df=session.table(TRAIN_DATA_TABLE+"_WEEKLY").to_pandas()
    test_df=session.table(TEST_DATA_TABLE+"_WEEKLY").to_pandas()   
    #train_df['Data']='train'
    #test_df['Data']='test'
    train_dates=train_df['Date'].unique()
    test_dates=test_df['Date'].unique()
    unique_dates=np.union1d(train_dates, test_dates)
    #unique_dates=train_dates+test_dates
    #df=pd.concat([train_df,test_df])
    #df = df.sort_values(by=['PARENT_DEALER_CODE_MODEL_FAMILY', 'Date']).reset_index(drop=True)
    #unique_dates = df['Date'].sort_values().unique()
    # Step 2: Generate Time_Index for each unique Date
    date_df = pd.DataFrame({'Date': unique_dates})
    date_df=date_df.sort_values(by="Date",ascending=True)
    date_df['Time_Index'] = range(len(date_df))
    
    min_date = date_df['Date'].min()
    date_df['Regularized_Date'] = date_df['Time_Index'].apply(lambda x: min_date + timedelta(days=x * 7))
    #df=pd.merge(df,date_df[['Date','Regularized_Date']],on="Date",how="left")
    train_df=pd.merge(train_df,date_df,on="Date",how="left")
    test_df=pd.merge(test_df,date_df,on="Date",how="left")
    write_to_snowflake(session,train_df,TRAIN_DATA_TABLE+"_WEEKLY_REGULARISED","overwrite")
    write_to_snowflake(session,test_df,TEST_DATA_TABLE+"_WEEKLY_REGULARISED","overwrite")
    return date_df


def get_week_label(dates):
    day_of_month = dates.day
    if 1 <= day_of_month <= 7:
        return 'FIRST_WEEK'
    elif 8 <= day_of_month <= 14:
        return 'SECOND_WEEK'
    elif 15 <= day_of_month <= 21:
        return 'THIRD_WEEK'
    elif 22 <= day_of_month <= 31:
        return 'FOURTH_WEEK'
    else:
        return 'UNKNOWN'

def get_number_of_days_in_week(week_start_date):
    day = week_start_date.day
    

    if day in [1, 8, 15]:
        return 7
    elif day == 22:
        # Find last day of the month
        next_month = week_start_date + relativedelta(months=1)
        first_of_next_month = next_month.replace(day=1)
        last_day_of_current_month = first_of_next_month - pd.Timedelta(days=1)
        return (last_day_of_current_month.day - 22 + 1)  # From 22nd to last day (inclusive)
    else:
        return None

def get_percent_marriagedays(marriage_day, total_marriage_days):
    if total_marriage_days==0:
        return 0
    else:
        return marriage_day/total_marriage_days
def create_bins_special_zero(column_name, combined_df, combined_labels):
    # 1. Separate out zero and nonzero values
    nonzero_series = combined_df.loc[
        (combined_df['Data'] == 'train') & 
        (combined_df[column_name] > 0), 
        column_name
    ]
    
    # 2. Apply qcut only on non-zero values
    _, bin_edges = pd.qcut(
        nonzero_series,
        q=len(combined_labels),
        labels=None,    # no labels yet
        retbins=True,
        duplicates='drop'
    )
    
    # 3. Adjust labels
    actual_num_bins = len(bin_edges) - 1
    adjusted_labels = combined_labels[:actual_num_bins]
    if actual_num_bins==2:
        adjusted_labels=['Low','High']

    return bin_edges, adjusted_labels

def create_bins(column_name,combined_df,combined_labels):
    _, bin_edges = pd.qcut(
        combined_df.loc[combined_df['Data'] == 'train', column_name],
        q=len(combined_labels),
        labels=None,
        retbins=True,
        duplicates='drop'
    )
    actual_num_bins = len(bin_edges) - 1
    adjusted_labels = combined_labels[:actual_num_bins]
    if actual_num_bins==2:
        adjusted_labels=['Low','High']
    
    return bin_edges,adjusted_labels
def assign_marriage_bin(x,bin_edges,adjusted_labels):
    if x == 0:
        return 'NO'
    else:
        return pd.cut(
            [x], 
            bins=bin_edges, 
            labels=adjusted_labels, 
            include_lowest=True
        )[0]
def create_dates(session,sales_data,max_end_period,train_end=TRAIN_END_DATE):
    grouped = sales_data.groupby(['PARENT_DEALER_CODE', 'MODEL_FAMILY','FAMILY_CODE'])
    results = []
    #print(len(grouped))
    for (parent_dealer_code, model_family,family_code), group in grouped:

        
        # Find the minimum date in the group
        start_date = min(group['Date'])
        
        # Generate all relevant dates
        dates = first_dates_of_months(start_date, max_end_period)
        
        # Create a DataFrame for this combination
        
        temp_df = pd.DataFrame({
            'Date': dates,
            'PARENT_DEALER_CODE': parent_dealer_code,
            'MODEL_FAMILY': model_family,
            'FAMILY_CODE':family_code
        })
        results.append(temp_df)
        
        
    results_df=pd.concat(results)   
    if results_df.shape[0]>0:
        festive_dates_month=session.table("WORK_DATABASE.MOP.FESTIVE_INDIAN_SEASON_AGG_MONTH").to_pandas()
        festive_dates_month=festive_dates_month.rename(columns={'MONTH_DATE':'Date'})
        temp_df=pd.merge(results_df,festive_dates_month,on="Date",how="left")
        final_data=pd.merge(temp_df,sales_data[['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE','Date','NET_SALES']],on=['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE','Date'],how="left")
        final_data.loc[pd.isnull(final_data['NET_SALES']),'NET_SALES']=0
        final_data['PARENT_DEALER_CODE_MODEL_FAMILY']=final_data.apply(lambda row:row['PARENT_DEALER_CODE']+"_"+row['MODEL_FAMILY']+"_"+row['FAMILY_CODE'],axis=1)
        final_data=final_data.fillna(0)
        train_df=final_data[final_data['Date']<train_end]
        test_df=final_data[final_data['Date']>=train_end]
        if train_df.shape[0]>0:
            write_to_snowflake(session,train_df,TRAIN_DATA_TABLE,"append")
            write_to_snowflake(session,test_df,TEST_DATA_TABLE,"append")
 
def appendFestive(session,table_name):
    query = f''' select * from  MOP_DATABASE.SOQ.FESTIVE_DAYS_SOQ '''
    festive_df =session.sql(query).to_pandas()

    data=session.table(table_name).to_pandas()
    merged_df = pd.merge(data,festive_df, left_on="Date", right_on="Date", how="left")

    return merged_df

def createWeeklyViews(session):
    train_df=session.table(TRAIN_DATA_TABLE+"_WEEKLY_REGULARISED").to_pandas()
    test_df=session.table(TEST_DATA_TABLE+"_WEEKLY_REGULARISED").to_pandas()
    drop_cols=['Next_Week_Date','NUMBER_OF_DAYS_WEEK','NUMBER_OF_DAYS_NEXT_WEEK','MONTH_DATE',"Date","Data"]
    train_df.drop(drop_cols,axis=1,inplace=True)
    test_df.drop(drop_cols,axis=1,inplace=True)

    updated_col_names=[col for col in train_df.columns if col.startswith("'")]
    new_col_name={col:col.replace("'","").replace(" ","_").upper() for col in updated_col_names}
    train_df=train_df.rename(columns=new_col_name)
    train_df=train_df.rename(columns={'Regularized_Date':"REGULARIZED_DATE"})
    test_df=test_df.rename(columns=new_col_name)
    test_df=test_df.rename(columns={'Regularized_Date':"REGULARIZED_DATE"})
    train_df.drop(['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE'],axis=1,inplace=True)
    test_df.drop(['PARENT_DEALER_CODE','MODEL_FAMILY','FAMILY_CODE','NET_SALES'],axis=1,inplace=True)
    
    write_to_snowflake(session,train_df,TRAIN_DATA_TABLE+"_WEEKLY_REGULARISED_FINAL","overwrite")

    write_to_snowflake(session,test_df,TEST_DATA_TABLE+"_WEEKLY_REGULARISED_FINAL","overwrite")
    return session.create_dataframe(train_df)
    
    

def main(session: snowpark.Session): 
    
    sku_supercedence=fetchSKUSupercedence(session)
    
    transit_data=fetchTransitData(session)
    parent_dealer_mapping=fetchParentDealerMapping(session)
    transit_data=pd.merge(transit_data,parent_dealer_mapping,on="DEALER_CODE")

    agg_data_sp=session.create_dataframe(transit_data)
    agg_data_sp.write.mode("overwrite").save_as_table("MOP_DATABASE.SOQ.TRANSIT_DEALER_MAPPING_NEW")
    
    
    sku_plant_mapping=fetchSkuPlantMapping(session)

    transit_data=pd.merge(transit_data,sku_plant_mapping,on='PLANT',how="inner")
    agg_data_sp=session.create_dataframe(transit_data)
    agg_data_sp.write.mode("overwrite").save_as_table("MOP_DATABASE.SOQ.TRANSIT_DEALER_SKU_PLANT_MAPPING_NEW")

    ### For each PLANT, SKU, PARENT DEALER GET THE MAX, min and MEAN TRANSIT TIME
    df=fetchAggregateTransitTime(session)

    df.create_or_replace_view("MOP_DATABASE.SOQ.PARENT_DEALER_TRANSIT_TIME_SKU_NEW")
    #return sku_supercedence

    #return agg_data_sp
    
    
    ecr_dealer_sales=ECRDealerSalesModelFamily(session,AGG_TYPE,CUSTOMER_TYPE_TO_CONSIDER,START_DATE)
    
    if AGG_TYPE=="weekly":
        session.create_dataframe(ecr_dealer_sales).write.mode("overwrite").save_as_table(ECR_DATA_TABLE+"_WEEKLY")
    else:
        session.create_dataframe(ecr_dealer_sales).write.mode("overwrite").save_as_table(ECR_DATA_TABLE)
        
    
    ecr_dealer_sales=getValidCombinations(session,ecr_dealer_sales,datetime.datetime(2024,3,1))

    ecr_dealer_sales['RUN_DATE']=datetime.datetime.now().strftime('%Y%m%d') 

    
    dealers=ecr_dealer_sales['PARENT_DEALER_CODE'].unique()

    
    if AGG_TYPE=="monthly":
        print("Number of Dealer ",len(dealers))
        #dealers=dealers[958:]
        for idx,dealer in enumerate(dealers):
            print(dealer)
            sales_data=ecr_dealer_sales[ecr_dealer_sales['PARENT_DEALER_CODE']==dealer]
            #print(sales_data.head())
            #create_test_dates(session,sales_data,MAX_DATE,TRAIN_END_DATE)
            create_dates(session,sales_data,MAX_DATE,TRAIN_END_DATE)
            if idx%50==0:
                print(idx)
        del ecr_dealer_sales
        train_merged_df=appendFestive(session,TRAIN_DATA_TABLE)
        write_to_snowflake(session,train_merged_df,TRAIN_DATA_TABLE+"_FESTIVE","overwrite")
    
        test_merged_df=appendFestive(session,TEST_DATA_TABLE)
        write_to_snowflake(session,test_merged_df,TEST_DATA_TABLE+"_FESTIVE","overwrite")
        #return session.create_dataframe(train_merged_df)
        
        return agg_data_sp
    else:
        
        print("number of Dealer ",len(dealers))
        
        for idx,dealer in enumerate(dealers):
            sales_data=ecr_dealer_sales[ecr_dealer_sales['PARENT_DEALER_CODE']==dealer]
            create_dates_weekly(session,sales_data,MAX_DATE,TRAIN_END_DATE)
        
            if idx%50==0:
                print(idx)
        dates_df=create_regularised_dates(session)
    
        #dates_df=create_regularised_dates(session)
        agg_data_sp=createWeeklyViews(session)
    
    return agg_data_sp

