import base64
import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\OneDrive\OneDrive - Tata Insights and Quants\Documents\travel_AA2\travel_AA2.csv')


df['Customer_Name_decoded'] = ''
df['Email_decoded'] = ''

df = df.astype(str)
def base_decode(inp_t):
    value_ascii = inp_t.encode("ascii")
    strbase64 = base64.b64decode(value_ascii)
    value_64 = strbase64.decode("ascii")
    value_64 = str(value_64).strip('}')
    return value_64

for index,row in df.iterrows():
    if not row['Customer Name (evar27)'] == 'nan': 
        row['Customer_Name_decoded'] = base_decode(row['Customer Name (evar27)'])
    if not row['Email (evar26)'] == 'nan': 
        row['Email_decoded'] = base_decode(row['Email (evar26)'])
    
    

df.to_csv(r'D:\OneDrive\OneDrive - Tata Insights and Quants\Documents\travel_AA2\travel_AA2_decoded.csv')

print(df.columns)