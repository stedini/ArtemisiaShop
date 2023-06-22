#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install lifetimes

# In[ ]:


import pandas as pd

# Load the data
data = pd.read_excel('Venduto Complessivo per cliente.xls')

# Display the first few rows of the data
data.head()

# In[ ]:


# Check for missing values
data.isnull().sum()

# In[ ]:


# Get descriptive statistics for the numerical columns
data.describe()

# In[ ]:


# Get unique values in the 'Cliente' column
data['Cliente'].nunique()

# In[ ]:


# Check the data type of the 'Giorno' column
data['Giorno'].dtype

# In[ ]:


# Check the data types of the columns
data.dtypes

# In[ ]:


# Check for missing values
data.isnull().sum()

# In[ ]:


# Count the number of unique clients
num_unique_clients = data['Cliente'].nunique()
num_unique_clients

# In[ ]:


# Calculate summary statistics for 'Venduto _Q' and 'Venduto _P'
data[['Venduto _Q', 'Venduto _P']].describe()

# In[ ]:


# Find the minimum and maximum date in the 'Giorno' column
min_date = data['Giorno'].min()
max_date = data['Giorno'].max()

min_date, max_date

# In[ ]:


# Import necessary libraries
from lifetimes.utils import summary_data_from_transaction_data
import datetime as dt

# Convert 'Giorno' to datetime
data['Giorno'] = pd.to_datetime(data['Giorno'])

# Get the end of the period under study
observation_period_end = data['Giorno'].max()

# Transform the data to RFM format
summary = summary_data_from_transaction_data(data, 'Cliente', 'Giorno', observation_period_end=observation_period_end)

# Display the first few rows of the summary data
summary.head()

# In[ ]:


# Import the BG/NBD model
from lifetimes import BetaGeoFitter

# Fit the BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])

# Print the model parameters
bgf

# In[ ]:


# Predict the expected number of purchases in the next 365 days
summary['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(365, summary['frequency'], summary['recency'], summary['T'])

# Sort customers by predicted purchases
top_customers = summary.sort_values(by='predicted_purchases', ascending=False)

# Select the top 100 customers
top_100_customers = top_customers.head(100)

# Display the top 100 customers
top_100_customers

# In[ ]:


# Save the output to a CSV file
summary.to_csv('BG_NBD_output_analysis.csv')

# In[ ]:


# Calculate the monetary value of the transactions for each customer
summary['monetary_value'] = data.groupby('Cliente')['Venduto _P'].mean()

# Check the correlation between frequency and monetary value
summary[['frequency', 'monetary_value']].corr()

# In[ ]:


# Import the Gamma-Gamma model
from lifetimes import GammaGammaFitter

# Remove transactions with negative monetary values
data_positive = data[data['Venduto _P'] > 0]

# Recalculate the monetary value of the transactions for each customer
summary_positive = summary_data_from_transaction_data(data_positive, 'Cliente', 'Giorno', monetary_value_col ='Venduto _P', observation_period_end=observation_period_end)

# Create a subset of the data that only includes customers who have made at least one repeat purchase
returning_customers_summary = summary[summary['frequency']>0]

returning_customers_summary_positive = returning_customers_summary[returning_customers_summary['monetary_value']>0]


# In[ ]:


# Fit the Gamma-Gamma model
ggf_positive = GammaGammaFitter(penalizer_coef = 0)
ggf_positive.fit(returning_customers_summary_positive['frequency'],
                 returning_customers_summary_positive['monetary_value'])

# Print the model parameters
ggf_positive

# In[ ]:


# Predict the average transaction value
returning_customers_summary_positive['predicted_avg_purchase'] = ggf_positive.conditional_expected_average_profit(
    returning_customers_summary_positive['frequency'],
    returning_customers_summary_positive['monetary_value']
)

# Predict the customer lifetime value
returning_customers_summary_positive['predicted_clv'] = ggf_positive.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    returning_customers_summary_positive['frequency'],
    returning_customers_summary_positive['recency'],
    returning_customers_summary_positive['T'],
    returning_customers_summary_positive['monetary_value'],
    time=6, # months
    discount_rate=0.01 # monthly discount rate ~ 12.7% annually
)

# Display the first few rows of the summary data
returning_customers_summary_positive.head()

# In[ ]:


# Display the full output data
returning_customers_summary_positive

# In[ ]:


# Define a function to assign each customer to a segment based on their predicted CLV
def assign_segment(row):
    if 'Generico Cliente' in row.name or 'NUOVO CLIENTE1' in row.name:
        return 'new'
    elif row['predicted_clv'] > 500:
        return 'high'
    elif 150 <= row['predicted_clv'] <= 500:
        return 'medium'
    else:
        return 'low'

# Apply the function to each row in the data
returning_customers_summary_positive['segment'] = returning_customers_summary_positive.apply(assign_segment, axis=1)

# Display the first few rows of the data
returning_customers_summary_positive

# In[ ]:


returning_customers_summary_positive.to_csv('CLV Analysis.csv')
