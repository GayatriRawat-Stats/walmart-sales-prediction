import numpy as np
import pandas as pd
sales = pd.read_csv(r'C:\Users\rawat\Downloads\archive (1).zip')
sales['Date'] = pd.to_datetime(sales['Date'], format='%d-%m-%Y')
print(sales.isnull().sum())
weekly_sales = sales.groupby(['Store','Date']).agg({'Weekly_Sales':'sum','Temperature':'mean','Fuel_Price':'mean','CPI':'mean','Unemployment':'mean','Holiday_Flag':'first'}).reset_index()
weekly_sales.to_csv('weekly_sales_cleaned.csv', index=False)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,6))
sns.lineplot(data=weekly_sales, x='Date', y='Weekly_Sales')
plt.title('Weekly Sales Over Time')
plt.show()
weekly_sales['Week'] = weekly_sales['Date'].dt.isocalendar().week
sns.boxplot(x='Week', y='Weekly_Sales', data=weekly_sales)
plt.title('Seasonality by Week Number')
plt.tight_layout()
plt.show()
sns.heatmap(weekly_sales[['Weekly_Sales','Temperature','Fuel_Price','CPI','Unemployment']].corr(), annot=True)
plt.show()

weekly_sales.sort_values(['Store','Date'], inplace=True)
weekly_sales['lag_1'] = weekly_sales.groupby(['Store'])['Weekly_Sales'].shift(1)
weekly_sales['rolling_mean_4'] = weekly_sales.groupby(['Store'])['Weekly_Sales'].transform(lambda x: x.rolling(4).mean())
weekly_sales['Holiday_Flag'] = weekly_sales['Holiday_Flag'].astype(int)
weekly_sales.dropna(inplace=True)

from statsmodels.tsa.statespace.sarimax import SARIMAX

exog_vars = ['Temperature','Fuel_Price','CPI','Unemployment','Holiday_Flag','lag_1','rolling_mean_4']

train = weekly_sales[weekly_sales['Date'] < '2012-10-26']
test = weekly_sales[weekly_sales['Date'] >= '2012-10-26']

model = SARIMAX(
    train['Weekly_Sales'],
    exog=train[exog_vars],
    order=(1,1,1),
    seasonal_order=(1,1,1,52)
)
results = model.fit()
forecast = results.predict(
    start=len(train),
    end=len(train)+len(test)-1,
    exog=test[exog_vars]
)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test['Weekly_Sales'], forecast)
print('MAE:', mae)
from prophet import Prophet

prophet_df = weekly_sales[['Date','Weekly_Sales']].rename(columns={'Date':'ds','Weekly_Sales':'y'})
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.add_country_holidays(country_name='US')
model.fit(prophet_df)

future = model.make_future_dataframe(periods=12, freq='W')
forecast = model.predict(future)
model.plot(forecast)
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(test['Weekly_Sales'], forecast))
print('RMSE:', rmse)
plt.figure(figsize=(12,6))
plt.plot(test['Date'], test['Weekly_Sales'], label='Actual')
plt.plot(test['Date'], forecast, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Weekly Sales')
plt.show()
