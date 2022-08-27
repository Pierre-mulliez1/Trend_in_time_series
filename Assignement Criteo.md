---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Title: Code for criteo assignment  
## Author: Pierre Mulliez
### Started on: 24-08-2022

```python
#Load dataset and import packages 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data.csv')
```

```python
#First glance
print('printing head: ')
print(df.head(5))
print('printing quant description: ')
print(df.describe())
```

```python
plt.figure(figsize=(15,10))
plt.plot(df.index,df.iloc[:,2:])
plt.legend(df.iloc[:,2:].columns)
plt.xlabel('Date', fontsize = 15)
x = [1,30,60,90,120,150,180]
label = ['01-01-2018','16-01-2018','01-02-2018','16-02-2018','01-03-2018','16-03-2018','31-03-2018']
plt.xticks(x, label,rotation = 45)
plt.ylabel('Units in Millions', fontsize = 15)
plt.show()
```

```python
plt.figure(figsize=(15,10))
plt.plot(df.index,df.loc[:,['paid_impressions','organic_page_views','organic_sales_revenue']])
plt.legend(['paid_impressions','organic_page_views','organic_sales_revenue'])
plt.xlabel('Date', fontsize = 15)
x = [1,30,60,90,120,150,180]
label = ['01-01-2018','16-01-2018','01-02-2018','16-02-2018','01-03-2018','16-03-2018','31-03-2018']
plt.xticks(x, label,rotation = 45)
plt.ylabel('Units in Millions', fontsize = 15)
plt.show()

plt.figure(figsize=(15,10))
plt.plot(df.index,df.loc[:,['paid_clicks','organic_sales_units','organic_transactions','paid_ad_revenue']])
plt.legend(['paid_clicks','organic_sales_units','organic_transactions','paid_ad_revenue'])
plt.xlabel('Date', fontsize = 15)
x = [1,30,60,90,120,150,180]
label = ['01-01-2018','16-01-2018','01-02-2018','16-02-2018','01-03-2018','16-03-2018','31-03-2018']
plt.xticks(x, label,rotation = 45)
plt.ylabel('Units in 10000', fontsize = 15)
plt.show()
```

```python
# find seasonality # organic sales view vs paid impression
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
plot_pacf(df['organic_sales_revenue'], lags=50, method="ywm")
plt.show()
dfdecompose = df.groupby('date').sum('organic_sales_revenue')[['organic_sales_revenue','paid_impressions']]
dfdecompose.index = range(0,90)
seasonal_decompose(dfdecompose['organic_sales_revenue'], model='additive',period = 7).plot()
```

```python
seasonal_decompose(dfdecompose['paid_impressions'], model='additive',period = 7).plot()
```

```python
corr_matrix = df.iloc[:,2:].corr()
corr_matrix.style.background_gradient(cmap='RdBu_r')
```

```python
# focus on retailer B
dfB = df.loc[df['retailer'] == 'B',:].drop(['retailer'], axis=1).set_index('date')
corr_matrix = dfB.corr()
corr_matrix.style.background_gradient(cmap='RdBu_r')
```

```python
# focus on retailer A
dfA = df.loc[df['retailer'] == 'A',:].drop(['retailer'], axis=1).set_index('date')
corr_matrix = dfA.corr()
corr_matrix.style.background_gradient(cmap='RdBu_r')
```

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from sklearn.metrics import mean_squared_error
# Retailer A 
train = dfA.loc[:'25/03/2018',['organic_sales_revenue', 'paid_clicks']]
test = dfA.loc['25/03/2018':,['organic_sales_revenue', 'paid_clicks']]
model= SARIMAX(train['organic_sales_revenue'], 
     order=(1,1,1), seasonal_order= (1,1,1,7),
    enforce_invertibility=True, enforce_stationarity=True)
results=model.fit()
forecast = results.forecast(steps=7)
rmse = np.sqrt(mean_squared_error(forecast, test['organic_sales_revenue']))
```

```python
dfA.tail()
dfB.tail()
print(rmse)
print(forecast)
test
```

```python
plot = dfA.loc['01/03/2018':,'organic_sales_revenue']
plt.figure(figsize=(15,10))
plt.plot(plot.index, plot, 'o-g')
plt.plot(plot.index[-7:], forecast, 'o-b')
plt.xlabel("Days March")
plt.ylabel("revenue in millions")
plt.xticks(rotation = 80, fontsize = 8)
plt.title("revenue over time without paid clicks added")
plt.legend(['Real revenue', 'Forecasted revenue'])
plt.show()
```

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from sklearn.metrics import mean_squared_error
# Retailer A 

model= SARIMAX(train['organic_sales_revenue'], exog=train['paid_clicks'],
     order=(1,2,1), seasonal_order= (1,0,1,7),
    enforce_invertibility=True, enforce_stationarity=True)
results=model.fit()
forecast = results.forecast(steps=7, exog = test['paid_clicks'])
rmse = np.sqrt(mean_squared_error(forecast, test['organic_sales_revenue']))

```

```python
print(rmse)
```

```python
plot = dfA.loc['01/03/2018':,'organic_sales_revenue']
plt.figure(figsize=(15,10))
plt.plot(plot.index, plot, 'o-g')
plt.plot(plot.index[-7:], forecast, 'o-b')
plt.xlabel("Days March")
plt.ylabel("revenue in millions")
plt.xticks(rotation = 80, fontsize = 8)
plt.title("revenue over time WITH paid clicks added")
plt.legend(['Real revenue', 'Forecasted revenue'])
plt.show()
```

```python
# Retailer B 
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from sklearn.metrics import mean_squared_error

train = dfB.loc[:'25/03/2018',['organic_sales_revenue', 'paid_clicks']]
test = dfB.loc['25/03/2018':,['organic_sales_revenue', 'paid_clicks']]

model= SARIMAX(train['organic_sales_revenue'], exog=train['paid_clicks'],
     order=(1,2,1), seasonal_order= (1,0,1,7),
    enforce_invertibility=True, enforce_stationarity=True)
results=model.fit()
forecast = results.forecast(steps=7, exog = test['paid_clicks'])
rmse = np.sqrt(mean_squared_error(forecast, test['organic_sales_revenue']))
```

```python
rmse
```

```python
plot = dfB.loc['01/03/2018':,'organic_sales_revenue']
plt.figure(figsize=(15,10))
plt.plot(plot.index, plot, 'o-g')
plt.plot(plot.index[-7:], forecast, 'o-b')
# set axis titles
plt.xlabel("Days March")
plt.ylabel("revenue in millions")
plt.xticks(rotation = 80, fontsize = 8)
plt.title("revenue over time WITH paid clicks added")
plt.legend(['Real revenue', 'Forecasted revenue'])
plt.show()
```

```python
# check for inconsistency in paid cliks and revenue 
## might reveal bugs 
df['prcClicksRev'] = df['organic_sales_revenue'] / df['paid_clicks'] *100
check = df[['date','retailer','organic_sales_revenue','paid_clicks','prcClicksRev']]\
.sort_values('organic_sales_revenue', axis = 0,ascending = False).head(40)
check.style.background_gradient(cmap='RdBu_r')

```

```python
check = df[['date','retailer','organic_sales_revenue','paid_clicks','prcClicksRev']]\
.sort_values('prcClicksRev', axis = 0,ascending = False).head(40)
check.style.background_gradient(cmap='RdBu_r')
```

```python
print(df.groupby('retailer').mean('prcClicksRev')['prcClicksRev'])
def group_range(x):
    return x.max() - x.min()
df_diff =df.groupby('date')['prcClicksRev'].apply(group_range).sort_values(axis = 0,ascending = False).head(20)
pd.DataFrame(df_diff).style.background_gradient(cmap='RdBu_r')
```

```python
# CHECK revenue vs impression number 
df['ADrevperImpression'] = df['paid_ad_revenue'] / df['paid_impressions'] *100
check = df[['date','retailer','paid_ad_revenue','paid_impressions','ADrevperImpression']]\
.sort_values('paid_ad_revenue', axis = 0,ascending = False)
check.head(40).style.background_gradient(cmap='RdBu_r')


```

```python
checkscatter = check['retailer'].replace('A', 'green')
checkscatter = checkscatter.replace('B', 'blue')
plt.figure(figsize=(15,8))
plt.scatter(check.index,check['ADrevperImpression'],marker="o",c = checkscatter)
x = [1,30,60,90,120,150,180]
label = ['01-01-2018','16-01-2018','01-02-2018','16-02-2018','01-03-2018','16-03-2018','31-03-2018']
plt.xticks(x, label,rotation = 45)
plt.title('Revenue per impression over time', fontsize = 15)
plt.ylabel('Revenue per impression', fontsize = 12)
plt.xlabel('Date', fontsize = 12)
plt.plot()

```
