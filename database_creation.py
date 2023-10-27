import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

df = pd.read_csv("Application_Data.csv")

conn = sqlite3.connect("credit_card_application.db")
cursor = conn.cursor()

df = pd.read_csv("Application_Data.csv")
df.drop(df.columns[df.nunique() == 1][0],axis=1,inplace=True)
df.drop('Total_Bad_Debt',axis=1,inplace=True)
df.columns = df.columns.str.strip()

for col in df.select_dtypes(object).columns:
    df[col] = df[col].str.strip()

def onehotencode(data: pd.DataFrame,col: str) -> pd.DataFrame:
    encoder = OneHotEncoder(drop='first',sparse_output=False,max_categories=10)
    encoded_data = encoder.fit_transform(data[[col]])
    encoded_data = pd.DataFrame(encoded_data,columns=encoder.get_feature_names_out())
    return encoded_data

onehotencode_cols = ['Applicant_Gender','Income_Type','Family_Status','Housing_Type','Job_Title']

for col in onehotencode_cols:
    encoded_data = onehotencode(df,col)
    df = pd.concat([df,encoded_data],axis=1)
    df.drop(col,axis=1,inplace=True)

def ordinal_encode(data: pd.DataFrame, col: str) -> pd.Series:
    encoder = OrdinalEncoder(categories=[['Lower secondary','Secondary / secondary special','Incomplete higher','Higher education','Academic degree']])
    data[col] = encoder.fit_transform(data[[col]])
    data[col] = data[col].astype(np.int64)
    return data[col]

df['Education_Type'] = ordinal_encode(df,'Education_Type')

df = df[['Total_Good_Debt', 'Applicant_Gender_M', 'Income_Type_Working',
       'Job_Title_Laborers', 'Family_Status_Married', 'Total_Income',
       'Total_Children', 'Years_of_Working', 'Applicant_Age',
       'Income_Type_State servant','Status']]

# Create schema for the credit application table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS CREDIT_APPLICATION (
        TOTAL_GOOD_DEBT FLOAT,
        APPLICANT_GENDER_M INT,
        INCOME_TYPE_WORKING INT,
        JOB_TITLE_LABORERS INT,
        FAMILY_STATUS_MARRIED int,
        TOTAL_INCOME FLOAT,
        TOTAL_CHILDREN INT,
        YEARS_OF_WORKING INT,
        APPLICANT_AGE INT,
        INCOME_TYPE_STATE_SERVANT INT,
        STATUS INT
    )
''')
conn.commit()

# for idx, row in df.iterrows():
#     sql = f"INSERT INTO CREDIT_APPLICATION VALUES {tuple(row)}"
#     cursor.execute(sql)
#     conn.commit()

# Add data from the dataframe to the database
df.to_sql('CREDIT_APPLICATION',conn,if_exists='replace',index=False)

# Close the database connection
conn.close()
