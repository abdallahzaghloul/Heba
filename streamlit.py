import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn  as sns
import plotly.express as px
import re
import joblib as jb
import sklearn

st.write ('This is Deployment for Customer Segmentation from bank in India (Real Data From Kaggle)')
st.write ('<h1 style="text-align:center;color:purple;"> Customer Segmentation Deployment</h1>' , unsafe_allow_html=True)

st.write('Take a Look At The Data')
df = pd.read_csv("bank_transactions.csv")
st.table(df.head())

df.drop(df.index[959987] , axis=0 , inplace=True)
df.dropna(inplace=True)
df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'] , dayfirst=True)
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'] , dayfirst=True)
df['Age'] = df['TransactionDate'].dt.year - df['CustomerDOB'].dt.year

indexAge = df[(df['Age'] <= 20)].index
df.drop(indexAge,inplace=True)

indexAge1 = df[(df['Age'] >= 100)].index
df.drop(indexAge1,inplace=True)

df['CustGender'].replace(['F'], 0 ,inplace=True)
df['CustGender'].replace(['M'], 1 ,inplace=True)

df['Pre_Recency'] = df ['TransactionDate'] 
df['FirstDateCustVisitBanck'] = df ['TransactionDate']

RFM_df = df.groupby('CustomerID').agg({'TransactionID':'count' , 'CustomerDOB':'median' , 'CustGender':'first' , 'CustLocation':'first' , 'CustAccountBalance':'mean' , 
                                       'TransactionAmount (INR)':'mean' ,'Age' :'first' , 'Pre_Recency':'max' , 'FirstDateCustVisitBanck':'min'})


RFM_df ['Recency'] = RFM_df ['Pre_Recency'] - RFM_df['FirstDateCustVisitBanck']

RFM_df['Recency']=RFM_df['Recency'].astype(str)

RFM_df ['Recency'] = RFM_df ['Recency'].apply(lambda x :re.search('\d+', x ).group())
RFM_df['Recency']=RFM_df['Recency'].astype(int)

RFM_df.rename(columns={"TransactionID":"Frequency"},inplace=True)

RFM_df.drop(columns=['CustomerDOB', 'CustLocation','Pre_Recency', 'FirstDateCustVisitBanck'], axis= 0 ,inplace=True)

st.write ('After Some Editing We Extract That')
RFM_df = RFM_df.sample(n=10000,random_state=0).reset_index(drop=True)
st.table(RFM_df.head())


fig = plt.figure(figsize= (10,5))
ax = df.groupby('CustGender')['TransactionAmount (INR)'].mean().plot(kind='bar',color=['pink','blue'])
st.pyplot(fig)

fig1 = plt.figure(figsize= (10,5))
ax = df.groupby('CustGender')['CustAccountBalance'].mean().plot(kind='bar',color=['pink','blue'])
plt.ylabel('Transaction Amount (INR)')
st.pyplot(fig1)

cust_acc_bal= st.slider ('Insert Customer Account Balance Please' ,0 , 15436498 , 2000 )
st.write('Customer Balance is', cust_acc_bal)


ave_tran_amount = st.slider ('Insert the Average Transaction Amount Please' ,0 , 107000 , 2000 )
st.write('Average Transaction Amount is', ave_tran_amount)

def predict( cust_acc_bal , ave_tran_amount ):
    RFM_df = pd.DataFrame(columns=jb.load('Banck_Segmentation.h5')[1:])
