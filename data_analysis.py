#!/usr/bin/env python
# coding: utf-8

# In[5]:






# In[6]:


import pandas as pd


# In[8]:


import os
print(os.getcwd())


# In[9]:


revenue_df = pd.read_csv('./data/revenue_data.csv')
expense_df = pd.read_csv('./data/expense_data.csv')
customer_df = pd.read_csv('./data/customer_transactions.csv')


# In[10]:


# Preview the data
print(revenue_df.head(), expense_df.head(), customer_df.head())


# In[11]:


# Convert date columns to datetime
revenue_df['date'] = pd.to_datetime(revenue_df['date'])
expense_df['date'] = pd.to_datetime(expense_df['date'])
customer_df['date'] = pd.to_datetime(customer_df['date'])


# In[12]:


# Check for nulls
print("Revenue Nulls:\n", revenue_df.isnull().sum(), "\n")
print("Expense Nulls:\n", expense_df.isnull().sum(), "\n")
print("Customer Nulls:\n", customer_df.isnull().sum(), "\n")


# In[14]:





# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np


# In[33]:


def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df


# In[34]:


revenue_df = clean_columns(revenue_df)
expense_df = clean_columns(expense_df)
customer_df = clean_columns(customer_df)

# Confirm cleaned column names
print("Revenue Columns:", revenue_df.columns.tolist())
print("Expense Columns:", expense_df.columns.tolist())
print("Customer Columns:", customer_df.columns.tolist())


# In[35]:


# Create 'year' and 'month' columns for grouping
revenue_df['year'] = revenue_df['date'].dt.year
revenue_df['month'] = revenue_df['date'].dt.month

expense_df['year'] = expense_df['date'].dt.year
expense_df['month'] = expense_df['date'].dt.month

customer_df['year'] = customer_df['date'].dt.year
customer_df['month'] = customer_df['date'].dt.month


# In[36]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[37]:


sns.set(style="whitegrid")


# In[38]:


revenue_df['year_month'] = revenue_df['date'].dt.to_period('M').astype(str)


# In[39]:


print(revenue_df.columns)


# In[40]:


monthly_revenue = revenue_df.groupby(['year_month', 'category'])['revenue'].sum().reset_index()


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_revenue, x='year_month', y='revenue', hue='category', marker='o')
plt.xticks(rotation=45)
plt.title("Monthly Revenue by Category")
plt.xlabel("Year-Month")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()


# In[41]:


# Convert date to datetime if not already
expense_df['date'] = pd.to_datetime(expense_df['date'])

# Create year_month column
expense_df['year_month'] = expense_df['date'].dt.to_period('M').astype(str)

# Group by year_month and type
monthly_expense = expense_df.groupby(['year_month', 'type'])['amount'].sum().reset_index()


# In[42]:


# Sum all expenses per month
total_monthly_expense = expense_df.groupby('year_month')['amount'].sum().reset_index()
total_monthly_expense.rename(columns={'amount': 'total_expense'}, inplace=True)


# In[43]:


plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_expense, x='year_month', y='amount', hue='type', marker='o')
plt.xticks(rotation=45)
plt.title("Monthly Expenses by Type")
plt.xlabel("Year-Month")
plt.ylabel("Amount")
plt.tight_layout()
plt.show()


# In[44]:


# Aggregate revenue if needed
monthly_total_revenue = revenue_df.groupby('year_month')['revenue'].sum().reset_index()

# Merge with expense data
rev_exp_df = pd.merge(monthly_total_revenue, total_monthly_expense, on='year_month')

# Calculate profit
rev_exp_df['profit'] = rev_exp_df['revenue'] - rev_exp_df['total_expense']


# In[45]:


plt.figure(figsize=(14, 6))
sns.lineplot(data=rev_exp_df, x='year_month', y='revenue', label='Revenue')
sns.lineplot(data=rev_exp_df, x='year_month', y='total_expense', label='Expense')
sns.lineplot(data=rev_exp_df, x='year_month', y='profit', label='Profit', linestyle='--')
plt.xticks(rotation=45)
plt.title("Revenue vs Expense vs Profit")
plt.xlabel("Year-Month")
plt.ylabel("Amount")
plt.legend()
plt.tight_layout()
plt.show()


# In[46]:


customer_df['date'] = pd.to_datetime(customer_df['date'])

# Create 'year_month' column
customer_df['year_month'] = customer_df['date'].dt.to_period('M').astype(str)

# Count unique customers per month
unique_customers = customer_df.groupby('year_month')['customer_id'].nunique().reset_index()
unique_customers.rename(columns={'customer_id': 'unique_customers'}, inplace=True)


# In[47]:


# Calculate total spend per customer per month
customer_spend = customer_df.groupby(['year_month', 'customer_id'])['amount'].sum().reset_index()

# Calculate average spend per customer per month
avg_spend_per_customer = customer_spend.groupby('year_month')['amount'].mean().reset_index()
avg_spend_per_customer.rename(columns={'amount': 'avg_spend_per_customer'}, inplace=True)


# In[48]:


# Calculate total spend per customer
customer_total_spend = customer_df.groupby('customer_id')['amount'].sum().reset_index()

# Identify top 10% of customers by spend
high_value_customers = customer_total_spend.nlargest(int(len(customer_total_spend) * 0.1), 'amount')

# Get their details
high_value_customer_ids = high_value_customers['customer_id']
high_value_customer_data = customer_df[customer_df['customer_id'].isin(high_value_customer_ids)]


# In[49]:


# Transaction volume per month
transactions_per_month = customer_df.groupby('year_month')['customer_id'].count().reset_index()
transactions_per_month.rename(columns={'customer_id': 'transaction_volume'}, inplace=True)

# Revenue per customer per month
revenue_per_customer = customer_spend.groupby('year_month')['amount'].mean().reset_index()
revenue_per_customer.rename(columns={'amount': 'revenue_per_customer'}, inplace=True)

# Plot transaction volume
plt.figure(figsize=(14, 6))
sns.lineplot(data=transactions_per_month, x='year_month', y='transaction_volume', marker='o', color='blue')
plt.title("Transaction Volume per Month")
plt.xlabel("Year-Month")
plt.ylabel("Transactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot average revenue per customer
plt.figure(figsize=(14, 6))
sns.lineplot(data=revenue_per_customer, x='year_month', y='revenue_per_customer', marker='o', color='orange')
plt.title("Average Revenue per Customer")
plt.xlabel("Year-Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[52]:





# In[53]:


from sklearn.linear_model import LinearRegression
import numpy as np


# In[54]:


# Prepare data (convert 'year_month' to numerical format, e.g., month index)
rev_exp_df['date_index'] = pd.to_datetime(rev_exp_df['year_month']).dt.strftime('%Y%m').astype(int)


# In[55]:


# Define features and target
X = rev_exp_df[['date_index']]  # Features: date index
y = rev_exp_df['revenue']  # Target: revenue


# In[57]:


 #Train-test split (optional but useful for evaluation)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# In[58]:


# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# In[60]:


# Predict future revenue (next 6 months)
future_months = pd.date_range(rev_exp_df['year_month'].max(), periods=7, freq='Me').strftime('%Y%m').astype(int)[1:]

future_X = pd.DataFrame(future_months, columns=['date_index'])
future_revenue = model.predict(future_X)


# In[63]:


# Generate future months with the 'ME' frequency
future_months = pd.date_range(rev_exp_df['year_month'].max(), periods=7, freq='ME').strftime('%Y%m').astype(int)[1:]


# In[64]:


# Combine results using 'ME' frequency for month-end dates
forecast_df = pd.DataFrame({
    'year_month': pd.date_range(rev_exp_df['year_month'].max(), periods=7, freq='ME')[1:],
    'predicted_revenue': future_revenue
})


# In[67]:


rev_exp_df['year_month'] = pd.to_datetime(rev_exp_df['year_month'], format='%Y-%m')
forecast_df['year_month'] = pd.to_datetime(forecast_df['year_month'], format='%Y-%m')


# In[68]:


plt.figure(figsize=(14, 6))
sns.lineplot(data=rev_exp_df, x='year_month', y='revenue', label='Actual Revenue')
sns.lineplot(data=forecast_df, x='year_month', y='predicted_revenue', label='Forecasted Revenue', linestyle='--', color='red')
plt.xticks(rotation=45)
plt.title("Revenue Forecast for the Next 6 Months")
plt.legend()
plt.show()


# In[69]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# In[70]:


# Prepare the data
X = np.array(range(len(rev_exp_df))).reshape(-1, 1)
y = rev_exp_df['revenue'].values


# In[71]:


# Apply polynomial transformation
poly = PolynomialFeatures(degree=3)  # Adjust degree based on your data
X_poly = poly.fit_transform(X)


# In[72]:


# Fit the model
model = LinearRegression()
model.fit(X_poly, y)


# In[73]:


# Predict future values
X_future = np.array(range(len(rev_exp_df), len(rev_exp_df) + 6)).reshape(-1, 1)
X_future_poly = poly.transform(X_future)
future_revenue = model.predict(X_future_poly)


# In[76]:


# Combine results into a DataFrame
forecast_df = pd.DataFrame({
   'year_month': pd.date_range(rev_exp_df['year_month'].max(), periods=7, freq='ME')[1:],
    'predicted_revenue': future_revenue
})


# In[78]:


# Plot
plt.figure(figsize=(14, 6))
sns.lineplot(data=rev_exp_df, x='year_month', y='revenue', label='Actual Revenue')
sns.lineplot(data=forecast_df, x='year_month', y='predicted_revenue', label='Forecasted Revenue', linestyle='--', color='red')
plt.xticks(rotation=45)
plt.title("Revenue Forecast for the Next 6 Months")
plt.legend()
plt.show()


# In[80]:





# In[81]:


from statsmodels.tsa.arima.model import ARIMA


# In[82]:


# Fit ARIMA model
model = ARIMA(rev_exp_df['revenue'], order=(5, 1, 0))  # (p, d, q) parameters to adjust
model_fit = model.fit()


# In[83]:


# Forecast for next 6 months
forecast = model_fit.forecast(steps=6)


# In[85]:


# Combine results into a DataFrame
forecast_df = pd.DataFrame({
    'year_month': pd.date_range(rev_exp_df['year_month'].max(), periods=7, freq='ME')[1:],
    'predicted_revenue': future_revenue
})


# In[86]:


# Plot
plt.figure(figsize=(14, 6))
sns.lineplot(data=rev_exp_df, x='year_month', y='revenue', label='Actual Revenue')
sns.lineplot(data=forecast_df, x='year_month', y='predicted_revenue', label='Forecasted Revenue', linestyle='--', color='red')
plt.xticks(rotation=45)
plt.title("Revenue Forecast for the Next 6 Months")
plt.legend()
plt.show()


# In[87]:





# In[89]:




# In[90]:


from prophet import Prophet


# In[92]:


rev_exp_df['year_month'] = pd.to_datetime(rev_exp_df['year_month'])


# In[93]:


# Prepare the dataframe for Prophet
prophet_df = rev_exp_df[['year_month', 'revenue']].rename(columns={
    'year_month': 'ds',
    'revenue': 'y'
})


# In[94]:


from prophet import Prophet


# In[95]:


model_prophet = Prophet()
model_prophet.fit(prophet_df)


# In[97]:


pd.date_range(start='2023-01-01', periods=6, freq='ME')  # Month End


# In[98]:


forecast = model_prophet.predict(future)


# In[102]:


import plotly.graph_objects as go


# In[104]:


import plotly.graph_objects as go


# In[106]:


from prophet.plot import plot_plotly


# In[108]:


from prophet.plot import plot_plotly
from prophet import Prophet
import plotly.graph_objects as go
import pandas as pd


# In[109]:


prophet_df = rev_exp_df[['year_month', 'revenue']].rename(columns={'year_month': 'ds', 'revenue': 'y'})


# In[112]:


import plotly.graph_objects as go
import prophet.plot
prophet.plot.go = go


# In[113]:


from prophet.plot import plot_plotly
from prophet import Prophet
import plotly.graph_objects as go
import prophet.plot  # You need this to patch it
prophet.plot.go = go  # Patch the missing variable manually


# In[114]:


# Prepare your data
prophet_df = rev_exp_df[['year_month', 'revenue']].rename(columns={'year_month': 'ds', 'revenue': 'y'})

# Fit model
model_prophet = Prophet()
model_prophet.fit(prophet_df)

# Make future dataframe and predict
future = model_prophet.make_future_dataframe(periods=6, freq='ME')
forecast = model_prophet.predict(future)

# Plot
plot_plotly(model_prophet, forecast)


# In[115]:


import matplotlib.pyplot as plt


# In[116]:


plt.figure(figsize=(14, 6))
plt.plot(model_prophet.history['ds'], model_prophet.history['y'], label='Actual Revenue')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Revenue', linestyle='--', color='red')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, color='red', label='Confidence Interval')
plt.title("Actual vs Forecasted Revenue")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.legend()
plt.grid(True)
plt.show()


# In[117]:


model_prophet.plot_components(forecast)


# In[118]:


model_prophet = Prophet(yearly_seasonality=True)
model_prophet.fit(prophet_df)


# In[119]:


model_prophet = Prophet()
model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model_prophet.fit(prophet_df)


# In[120]:


# Example: Replace extreme values with rolling average
prophet_df['y'] = prophet_df['y'].rolling(window=3, center=True, min_periods=1).mean()


# In[121]:


prophet_df['y'] = np.log(prophet_df['y'])


# In[122]:


forecast['yhat'] = np.exp(forecast['yhat'])


# In[123]:


# Ensure all revenue values are positive
prophet_df = prophet_df[prophet_df['y'] > 0].copy()

# Apply log transformation
prophet_df['y'] = np.log(prophet_df['y'])


# In[124]:


model_prophet = Prophet(yearly_seasonality=True)
model_prophet.fit(prophet_df)

future = model_prophet.make_future_dataframe(periods=6, freq='ME')
forecast = model_prophet.predict(future)


# In[125]:


forecast['yhat'] = np.exp(forecast['yhat'])
forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])


# In[126]:


forecast['yhat'].max()


# In[127]:


forecast[['yhat', 'yhat_lower', 'yhat_upper']].max()


# In[128]:


import matplotlib.pyplot as plt


# In[129]:


# Log-transformed
plt.plot(prophet_df['ds'], prophet_df['y'], label='Log Revenue')
plt.title("Log-Transformed Revenue Over Time")
plt.show()


# In[130]:


# Raw revenue
plt.plot(prophet_df['ds'], np.exp(prophet_df['y']), label='Actual Revenue')
plt.title("Actual Revenue Over Time")
plt.show()


# In[131]:


print(np.exp(prophet_df['y'].tail()))


# In[132]:


prophet_df = pd.DataFrame({
    'ds': rev_exp_df['year_month'],
    'y': np.log(rev_exp_df['revenue'])  # this ensures log-transformation
})


# In[133]:


import numpy as np


# In[134]:


# Check raw revenue values
print("Raw revenue:", rev_exp_df['revenue'].tail().tolist())


# In[135]:


# Check logged values
print("Log-transformed:", np.log(rev_exp_df['revenue'].tail()).tolist())


# In[136]:


# Compare with what's in prophet_df
print("Prophet 'y' values:", prophet_df['y'].tail().tolist())


# In[137]:


prophet_df = pd.DataFrame({
    'ds': rev_exp_df['year_month'],
    'y': np.log(rev_exp_df['revenue'])  # log transformation
})
model_prophet = Prophet()
model_prophet.fit(prophet_df)


# In[139]:


future = model_prophet.make_future_dataframe(periods=6, freq='Me')
forecast = model_prophet.predict(future)


# In[140]:


future = model_prophet.make_future_dataframe(periods=6, freq='ME')  # Capital 'E' for month-end
forecast = model_prophet.predict(future)


# In[141]:


forecast_exp = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
forecast_exp[['yhat', 'yhat_lower', 'yhat_upper']] = np.exp(forecast_exp[['yhat', 'yhat_lower', 'yhat_upper']])


# In[142]:


import matplotlib.pyplot as plt


# In[144]:


 #Plot actual revenue
plt.figure(figsize=(14, 6))
plt.plot(rev_exp_df['year_month'], rev_exp_df['revenue'], label='Actual Revenue')

# Plot forecasted revenue
plt.plot(forecast_exp['ds'], forecast_exp['yhat'], label='Forecasted Revenue', linestyle='--', color='red')
plt.fill_between(forecast_exp['ds'], forecast_exp['yhat_lower'], forecast_exp['yhat_upper'], color='pink', alpha=0.3)

plt.title("Revenue Forecast (Back-transformed from log scale)")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[145]:


import matplotlib.pyplot as plt


# In[146]:


# Merge actuals and forecasts
forecast_df = forecast[['ds', 'yhat']].copy()
forecast_df = forecast_df.set_index('ds')
actual_df = prophet_df[['ds', 'y']].copy()
actual_df = actual_df.set_index('ds')


# In[147]:


# Convert from log scale back to original if needed
forecast_df['yhat'] = np.exp(forecast_df['yhat'])
actual_df['y'] = np.exp(actual_df['y'])


# In[148]:


# Plot
plt.figure(figsize=(14,6))
plt.plot(actual_df.index, actual_df['y'], label='Actual Revenue')
plt.plot(forecast_df.index, forecast_df['yhat'], label='Forecasted Revenue', linestyle='--', color='red')
plt.title('Actual vs Forecasted Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[149]:


import matplotlib.pyplot as plt


# In[150]:


# Merge actual (transformed back) and forecast
forecast_df = forecast[['ds', 'yhat']].copy()
forecast_df['yhat'] = np.exp(forecast_df['yhat'])


# In[151]:


# Actual values (from original data)
actual_df = prophet_df.copy()
actual_df['y'] = np.exp(actual_df['y'])


# In[152]:


# Plot
plt.figure(figsize=(12, 6))
plt.plot(actual_df['ds'], actual_df['y'], label='Actual Revenue', marker='o')
plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecasted Revenue', linestyle='--', color='red')
plt.title('Actual vs Forecasted Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[153]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# In[154]:


# Merge actual and forecasted (only for known periods)
merged_df = pd.merge(actual_df, forecast_df, on='ds', how='inner')
y_true = merged_df['y']
y_pred = merged_df['yhat']


# In[155]:


mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))


# In[156]:


print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


# In[158]:


# Make the forecast including uncertainty intervals
forecast = model_prophet.predict(future, uncertainty_samples=True)


# In[159]:


from prophet import Prophet


# In[160]:


# Set the interval width when initializing the Prophet model
model_prophet = Prophet(interval_width=0.95)  # 95% confidence interval


# In[161]:


# Fit the model
model_prophet.fit(prophet_df)


# In[162]:


# Make the forecast
future = model_prophet.make_future_dataframe(periods=6, freq='M')
forecast = model_prophet.predict(future)


# In[163]:


# Make the forecast with updated frequency
future = model_prophet.make_future_dataframe(periods=6, freq='ME')
forecast = model_prophet.predict(future)


# In[164]:


# Now, you should have 'yhat_lower' and 'yhat_upper' columns
forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# In[165]:


# Export the forecasted results
forecast_df.to_csv('forecasted_revenue.csv', index=False)


# In[ ]:


## Select only relevant columns
forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_df.columns = ['date', 'predicted_revenue', 'lower_bound', 'upper_bound']
#Clean and Prepare Forecast Data


# In[166]:


merged_df.to_csv('forecasted_revenue.csv', index=False)
print("Forecast exported successfully!")


# In[168]:


import matplotlib.pyplot as plt


# In[169]:


plt.figure(figsize=(14, 6))
plt.plot(merged_df['date'], np.exp(merged_df['actual_revenue']), label='Actual Revenue', marker='o')
plt.plot(merged_df['date'], np.exp(merged_df['predicted_revenue']), label='Forecasted Revenue', linestyle='--', color='red')
plt.fill_between(merged_df['date'],
                 np.exp(merged_df['lower_bound']),
                 np.exp(merged_df['upper_bound']),
                 color='pink', alpha=0.3, label='Uncertainty Interval')
plt.xticks(rotation=45)
plt.title("Actual vs Forecasted Revenue")
plt.legend()
plt.tight_layout()
plt.show()


# In[170]:


print(merged_df.columns)


# In[171]:


future = model_prophet.make_future_dataframe(periods=6, freq='M')
forecast = model_prophet.predict(future)


# In[172]:


future = model_prophet.make_future_dataframe(periods=6, freq='ME')


# In[173]:


forecast = model_prophet.predict(future)


# In[175]:


merged_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
merged_df = merged_df.merge(prophet_df.rename(columns={'ds': 'ds', 'y': 'actual_revenue'}), on='ds', how='left')
merged_df.rename(columns={'ds': 'date', 'yhat': 'predicted_revenue', 
                          'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'}, inplace=True)


# In[176]:


plt.figure(figsize=(14, 6))
plt.plot(merged_df['date'], np.exp(merged_df['actual_revenue']), label='Actual Revenue', marker='o')
plt.plot(merged_df['date'], np.exp(merged_df['predicted_revenue']), label='Forecasted Revenue', linestyle='--', color='red')
plt.fill_between(merged_df['date'],
                 np.exp(merged_df['lower_bound']),
                 np.exp(merged_df['upper_bound']),
                 color='pink', alpha=0.3, label='Uncertainty Interval')
plt.xticks(rotation=45)
plt.title("Actual vs Forecasted Revenue")
plt.legend()
plt.tight_layout()
plt.show()


# In[177]:





# In[178]:


import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt


# In[183]:


import os
print(os.getcwd())


# In[186]:


import os
print(os.listdir('/Users/neemaurassa/Business-performance-insights'))


# In[187]:


df = pd.read_csv('/Users/neemaurassa/Business-performance-insights/forecasted_revenue.csv')


# In[188]:


df['ds'] = pd.to_datetime(df['ds'])        # Ensure datetime format
df['y'] = np.log(df['y'])                  # Apply log if used in training


# In[189]:


# Prophet model
model = Prophet()
model.fit(df)


# In[190]:


# Future predictions
future = model.make_future_dataframe(periods=6, freq='ME')
forecast = model.predict(future)


# In[191]:


# Inverse transform to get actual revenue
forecast['yhat'] = np.exp(forecast['yhat'])
forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])


# In[192]:


# UI layout
st.title("📈 Revenue Forecast Dashboard")
st.write("This dashboard shows the forecasted revenue for the next 6 months using Prophet.")


# In[193]:





# In[194]:




# In[195]:



# In[1]:





# In[ ]:





