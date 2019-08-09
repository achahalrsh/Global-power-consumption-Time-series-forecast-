
import pandas as pd
import numpy as np
from matplotlib import pyplot
import seaborn as sns
import xgboost as xg
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tsfresh import extract_features #used to extract time series features
from sklearn import datasets
#from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
import math


# As we have concatenated all the csv files into a single "final.csv" without headers, 
# we define column names here to be used in our program later while training after inducing other features using tsfresh
# . The dataset is about 90MB in size and consists of all the observations made in the given time frame.

# In[2]:


data1 = pd.read_csv('final.csv', sep=',', low_memory=False,header=0, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'], na_values = '?')
data= data1
data.head()


# On inspecting the data file , we found the data is separated by comma(,) , so we use sep in the read_csv function to create a dataframe . Above shown are the first 5 rows of the dataset including headers(attributes). Also as we have '?' in the missing values of the data , so we include low_memory = False to avoid the error.
# 
# Consider an observation below, we have inconsistency in the data. We include na_values ='?' so that we can convert our data to float.
# 
# 2008-01-13,19:00:00,?,?,?,?,?,?,
# 
# Parsing the data so that its easier for python to infer date and time, and making the date and time collectively as index of our data, we do the necessary changes.

# For missing values , we replace them with the value of the previous day as below.

# In[539]:


def missing_vals(values):
    one_day = 60 * 24
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                    values[i, j] += values[i - one_day, j]
                


# In[540]:


missing_vals(data.values)


# To check the minute level spread of the data , we plot the given dataset as shownbelow. We see a seasonal effect in sub_meetering_3 which corresponds to electric heater and air conditioner energy.

# In[541]:


months = [x for x in range(1, 13)]
pyplot.figure()
for i in range(len(months)):
    ax = pyplot.subplot(len(months), 1, i+1)
    month = '2008-' + str(months[i])
    result = data[month]
    pyplot.plot(result['Global_active_power'])
    pyplot.title(month, y=0, loc='left')
pyplot.show()


# Now we can check the monthwise distribution of power active , spread across days. It shows that there are regular ups and downs in the active power trend which suits with the regular lifestyle of an average person.

# In[542]:


pyplot.figure()
for i in range(len(data.columns)):
    pyplot.subplot(len(data.columns), 1, i+1)
    name = data.columns[i]
    data[name].hist(bins=100)
    pyplot.title(name, y=0)
pyplot.show()


# We can also check the individual distribution of the attributes. Voltage seems to have a gaussian distribution whereas rest of the data seems skewed.

# Now moving onto the modelling part, as we are interested in predicting the per day power consumption for next 7 days, we can group the data on day level. This creates a data set on date level. We can also convert it to week level if we want to predict the weekly consumption of power ,but I am keeping it at day level as we need to predict it for next 7 days (day level).

# In[543]:


data = data.resample('D').sum()
print(data.head())


# We can plot a heatmap to visually compare the correlation amongst the attributes , which helps us to categorize important features and redundant features. "Independent variables or attributes" having high correlation doesn't add much value to our model , while increasing the unwanted dimension. Here in the plot below we can see that our output variable "Global_active_power" is most sensitive to GLobal_intensity as it is highly correlated with it, with a correlation factor of 1.
# 
# In similar manner , the second most sensitive variable to "Global_active_power" is Sub_metering_3.

# In[606]:




Var_Corr = data.corr()
# plot the heatmap and annotation on it
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)


# As , we are interested in predicting the Global active power consumption for the next day , we are introducing lagged features to assist our model learn the data better.Here we create 6 additional features having Global power of last 6 days in each column as an independent attribute.
# 

# In[545]:


df1 = data['Global_active_power']

df2 = pd.concat([df1.shift(6),df1.shift(5),df1.shift(4),df1.shift(3), df1.shift(2), df1.shift(1)], axis=1)
df3 =pd.concat([df2,data],axis =1)
df3.columns = ['power(t6)','power(t5)','power(t4)','power(t3)','power(t2)','power(t1)','Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
df3.head()


# Similarly, we introduce lead variables which will act as our target variables representing Global power consumption on each next day from day 1 to day 7.
# 

# In[546]:


df4 = pd.concat([df3,df1.shift(-1),df1.shift(-2),df1.shift(-3),df1.shift(-4),df1.shift(-5),df1.shift(-6),df1.shift(-7)],axis = 1)
df4.columns = ['power(t6)','power(t5)','power(t4)','power(t3)','power(t2)','power(t1)','Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3','active_power_t+1','active_power_t+2','active_power_t+3','active_power_t+4','active_power_t+5','active_power_t+6','active_power_t+7']
df4.head()


# In[547]:


#removing Nan values created
df4 = df4.iloc[6:]
df5 = df4.iloc[:-7]
df5.tail(7)


# In[ ]:


#adding few more variables
df5['sub_metering_4'] = df5['Global_active_power']*1000/60 - (df5['Sub_metering_1']+df5['Sub_metering_2']+df5['Sub_metering_3'])
df5['average_7_days'] = (df5['power(t6)']+df5['power(t5)']+df5['power(t4)']+df5['power(t3)']+df5['power(t2)']+df5['power(t1)']+df5['Global_active_power'])/7
df5['max_week_power'] = (df5[['power(t6)','power(t5)','power(t4)','power(t3)','power(t2)','power(t1)','Global_active_power']]).max(axis=1)
df5['min_week_power'] = (df5[['power(t6)','power(t5)','power(t4)','power(t3)','power(t2)','power(t1)','Global_active_power']]).min(axis=1)
df5['rate_of_change_7'] = (df5['Global_active_power']- df5['power(t6)'])/df5['power(t6)']
df5['rate_of_change_3'] = (df5['Global_active_power']- df5['power(t3)'])/(df5['power(t3)']+0.00001)
df5['Weekend_or_not'] = ((pd.DatetimeIndex(df5.index).dayofweek) // 5 == 1).astype('float64')


# New features like weekend and weekdays were added as active power cosumed changes very much for weekdays and weekend.
# 
# Few other features were also added like: Maximum power consumed during week, Minimum power consumed during week, rate of change of power consumed in a week etc.

# We use tsfresh library to generate more time series variables for our model. We create index as new_id and time as new_time column to our dataframe to facilitate the tsfresh library usage.

# In[ ]:


l=[]
for i in range(1061):
    for j in range(24*60):
        l.append(i)
for i in range(1527840,1527663,-1):
    l.pop(-1)
data1.insert(0,'new_id',l)

p=[]
for i in range(1061):
     for j in range(24*60):
         p.append(j)
for i in range(1527840,1527663,-1):
    p.pop(-1)
    
data1.insert(0,'new_time',p)


data_tsh=data1.iloc[:-1]
data_tsh=data_tsh[['new_time','new_id','Global_active_power']]

data_tsh['Global_active_power']=data_tsh['Global_active_power'].fillna(0)


extracted_features = extract_features(data_tsh, column_id="new_id", column_sort="new_time")

Features_db = extracted_features.dropna(axis='columns')
#remove constant columns
Features_db = Features_db.loc[:, Features_db.var() != 0.0]
Features_db1 = Features_db.iloc[6:]
Features_db1 = Features_db1[:-7]
Features_db1.insert(0, 'id', range(0, len(Features_db1)))
Features_db1.head()

#Creating a column id in df5 to merge with Features_db1 
df5.insert(0, 'id', range(0, len(df5)))
data_final = pd.merge(df5, Features_db1, on='id')


# We create two different list for holding variables in independent variables (attributes) and dependent variables(label)

# In[585]:


imp_train_cols = ['power(t6)','power(t5)','power(t4)','power(t3)','power(t2)','power(t1)','Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3','sub_metering_4','average_7_days','max_week_power','min_week_power','Weekend_or_not','Global_active_power__abs_energy', 'Global_active_power__absolute_sum_of_changes', 'Global_active_power__agg_autocorrelation__f_agg_"mean"__maxlag_40', 'Global_active_power__agg_autocorrelation__f_agg_"median"__maxlag_40', 'Global_active_power__agg_autocorrelation__f_agg_"var"__maxlag_40', 'Global_active_power__agg_linear_trend__f_agg_"max"__chunk_len_10__attr_"intercept"', 'Global_active_power__agg_linear_trend__f_agg_"max"__chunk_len_10__attr_"rvalue"', 'Global_active_power__agg_linear_trend__f_agg_"max"__chunk_len_10__attr_"slope"', 'Global_active_power__agg_linear_trend__f_agg_"max"__chunk_len_10__attr_"stderr"', 'Global_active_power__agg_linear_trend__f_agg_"max"__chunk_len_50__attr_"intercept"', 'Global_active_power__agg_linear_trend__f_agg_"max"__chunk_len_50__attr_"rvalue"', 'Global_active_power__agg_linear_trend__f_agg_"max"__chunk_len_50__attr_"slope"', 'Global_active_power__agg_linear_trend__f_agg_"max"__chunk_len_50__attr_"stderr"', 'Global_active_power__agg_linear_trend__f_agg_"max"__chunk_len_5__attr_"intercept"', 'Global_active_power__agg_linear_trend__f_agg_"max"__chunk_len_5__attr_"rvalue"', 'Global_active_power__agg_linear_trend__f_agg_"max"__chunk_len_5__attr_"slope"', 'Global_active_power__agg_linear_trend__f_agg_"max"__chunk_len_5__attr_"stderr"', 'Global_active_power__agg_linear_trend__f_agg_"mean"__chunk_len_10__attr_"intercept"', 'Global_active_power__agg_linear_trend__f_agg_"mean"__chunk_len_10__attr_"rvalue"', 'Global_active_power__agg_linear_trend__f_agg_"mean"__chunk_len_10__attr_"slope"', 'Global_active_power__agg_linear_trend__f_agg_"mean"__chunk_len_10__attr_"stderr"', 'Global_active_power__agg_linear_trend__f_agg_"mean"__chunk_len_50__attr_"intercept"', 'Global_active_power__agg_linear_trend__f_agg_"mean"__chunk_len_50__attr_"rvalue"', 'Global_active_power__agg_linear_trend__f_agg_"mean"__chunk_len_50__attr_"slope"', 'Global_active_power__agg_linear_trend__f_agg_"mean"__chunk_len_50__attr_"stderr"', 'Global_active_power__agg_linear_trend__f_agg_"mean"__chunk_len_5__attr_"intercept"', 'Global_active_power__agg_linear_trend__f_agg_"mean"__chunk_len_5__attr_"rvalue"', 'Global_active_power__agg_linear_trend__f_agg_"mean"__chunk_len_5__attr_"slope"', 'Global_active_power__agg_linear_trend__f_agg_"mean"__chunk_len_5__attr_"stderr"', 'Global_active_power__agg_linear_trend__f_agg_"min"__chunk_len_10__attr_"intercept"', 'Global_active_power__agg_linear_trend__f_agg_"min"__chunk_len_10__attr_"rvalue"', 'Global_active_power__agg_linear_trend__f_agg_"min"__chunk_len_10__attr_"slope"', 'Global_active_power__agg_linear_trend__f_agg_"min"__chunk_len_10__attr_"stderr"', 'Global_active_power__agg_linear_trend__f_agg_"min"__chunk_len_50__attr_"intercept"', 'Global_active_power__agg_linear_trend__f_agg_"min"__chunk_len_50__attr_"rvalue"', 'Global_active_power__agg_linear_trend__f_agg_"min"__chunk_len_50__attr_"slope"', 'Global_active_power__agg_linear_trend__f_agg_"min"__chunk_len_50__attr_"stderr"', 'Global_active_power__agg_linear_trend__f_agg_"min"__chunk_len_5__attr_"intercept"', 'Global_active_power__agg_linear_trend__f_agg_"min"__chunk_len_5__attr_"rvalue"', 'Global_active_power__agg_linear_trend__f_agg_"min"__chunk_len_5__attr_"slope"', 'Global_active_power__agg_linear_trend__f_agg_"min"__chunk_len_5__attr_"stderr"', 'Global_active_power__agg_linear_trend__f_agg_"var"__chunk_len_10__attr_"intercept"', 'Global_active_power__agg_linear_trend__f_agg_"var"__chunk_len_10__attr_"rvalue"', 'Global_active_power__agg_linear_trend__f_agg_"var"__chunk_len_10__attr_"slope"', 'Global_active_power__agg_linear_trend__f_agg_"var"__chunk_len_10__attr_"stderr"', 'Global_active_power__agg_linear_trend__f_agg_"var"__chunk_len_50__attr_"intercept"', 'Global_active_power__agg_linear_trend__f_agg_"var"__chunk_len_50__attr_"rvalue"', 'Global_active_power__agg_linear_trend__f_agg_"var"__chunk_len_50__attr_"slope"', 'Global_active_power__agg_linear_trend__f_agg_"var"__chunk_len_50__attr_"stderr"', 'Global_active_power__agg_linear_trend__f_agg_"var"__chunk_len_5__attr_"intercept"', 'Global_active_power__agg_linear_trend__f_agg_"var"__chunk_len_5__attr_"rvalue"', 'Global_active_power__agg_linear_trend__f_agg_"var"__chunk_len_5__attr_"slope"', 'Global_active_power__agg_linear_trend__f_agg_"var"__chunk_len_5__attr_"stderr"', 'Global_active_power__approximate_entropy__m_2__r_0.1', 'Global_active_power__approximate_entropy__m_2__r_0.3', 'Global_active_power__approximate_entropy__m_2__r_0.5', 'Global_active_power__approximate_entropy__m_2__r_0.7', 'Global_active_power__approximate_entropy__m_2__r_0.9', 'Global_active_power__ar_coefficient__k_10__coeff_0', 'Global_active_power__ar_coefficient__k_10__coeff_1', 'Global_active_power__ar_coefficient__k_10__coeff_2', 'Global_active_power__ar_coefficient__k_10__coeff_3', 'Global_active_power__ar_coefficient__k_10__coeff_4', 'Global_active_power__augmented_dickey_fuller__attr_"usedlag"', 'Global_active_power__binned_entropy__max_bins_10', 'Global_active_power__c3__lag_1', 'Global_active_power__c3__lag_2', 'Global_active_power__c3__lag_3', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_0.2__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.4', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.4', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.6', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.4', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.4', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6', 'Global_active_power__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_0.2__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.4', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.6', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.4', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_0.2__ql_0.0' ,'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.4', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.4', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.6', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.0', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.2', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6', 'Global_active_power__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8', 'Global_active_power__cid_ce__normalize_False', 'Global_active_power__cid_ce__normalize_True', 'Global_active_power__count_above_mean', 'Global_active_power__count_below_mean', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_0__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_0__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_0__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_0__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_10__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_10__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_10__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_10__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_11__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_11__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_11__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_11__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_12__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_12__w_2'
, 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_12__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_12__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_13__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_13__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_13__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_13__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_14__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_14__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_14__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_14__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_1__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_1__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_1__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_1__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_2__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_2__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_2__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_2__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_3__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_3__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_3__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_3__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_4__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_4__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_4__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_4__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_5__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_5__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_5__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_5__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_6__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_6__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_6__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_6__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_7__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_7__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_7__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_7__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_8__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_8__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_8__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_8__w_5', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_9__w_10', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_9__w_2', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_9__w_20', 'Global_active_power__cwt_coefficients__widths_(2, 5, 10, 20)__coeff_9__w_5', 'Global_active_power__fft_coefficient__coeff_0__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_0__attr_"real"', 'Global_active_power__fft_coefficient__coeff_10__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_10__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_10__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_10__attr_"real"', 'Global_active_power__fft_coefficient__coeff_11__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_11__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_11__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_11__attr_"real"', 'Global_active_power__fft_coefficient__coeff_12__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_12__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_12__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_12__attr_"real"', 'Global_active_power__fft_coefficient__coeff_13__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_13__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_13__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_13__attr_"real"', 'Global_active_power__fft_coefficient__coeff_14__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_14__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_14__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_14__attr_"real"', 'Global_active_power__fft_coefficient__coeff_15__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_15__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_15__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_15__attr_"real"', 'Global_active_power__fft_coefficient__coeff_16__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_16__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_16__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_16__attr_"real"', 'Global_active_power__fft_coefficient__coeff_17__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_17__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_17__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_17__attr_"real"', 'Global_active_power__fft_coefficient__coeff_18__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_18__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_18__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_18__attr_"real"', 'Global_active_power__fft_coefficient__coeff_19__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_19__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_19__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_19__attr_"real"', 'Global_active_power__fft_coefficient__coeff_1__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_1__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_1__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_1__attr_"real"', 'Global_active_power__fft_coefficient__coeff_20__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_20__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_20__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_20__attr_"real"', 'Global_active_power__fft_coefficient__coeff_21__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_21__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_21__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_21__attr_"real"', 'Global_active_power__fft_coefficient__coeff_22__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_22__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_22__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_22__attr_"real"', 'Global_active_power__fft_coefficient__coeff_23__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_23__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_23__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_23__attr_"real"', 'Global_active_power__fft_coefficient__coeff_24__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_24__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_24__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_24__attr_"real"', 'Global_active_power__fft_coefficient__coeff_25__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_25__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_25__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_25__attr_"real"', 'Global_active_power__fft_coefficient__coeff_26__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_26__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_26__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_26__attr_"real"', 'Global_active_power__fft_coefficient__coeff_27__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_27__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_27__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_27__attr_"real"', 'Global_active_power__fft_coefficient__coeff_28__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_28__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_28__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_28__attr_"real"', 'Global_active_power__fft_coefficient__coeff_29__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_29__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_29__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_29__attr_"real"', 'Global_active_power__fft_coefficient__coeff_2__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_2__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_2__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_2__attr_"real"', 'Global_active_power__fft_coefficient__coeff_30__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_30__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_30__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_30__attr_"real"', 'Global_active_power__fft_coefficient__coeff_31__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_31__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_31__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_31__attr_"real"', 'Global_active_power__fft_coefficient__coeff_32__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_32__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_32__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_32__attr_"real"', 'Global_active_power__fft_coefficient__coeff_33__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_33__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_33__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_33__attr_"real"', 'Global_active_power__fft_coefficient__coeff_34__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_34__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_34__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_34__attr_"real"', 'Global_active_power__fft_coefficient__coeff_35__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_35__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_35__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_35__attr_"real"', 'Global_active_power__fft_coefficient__coeff_36__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_36__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_36__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_36__attr_"real"', 'Global_active_power__fft_coefficient__coeff_37__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_37__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_37__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_37__attr_"real"', 'Global_active_power__fft_coefficient__coeff_38__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_38__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_38__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_38__attr_"real"', 'Global_active_power__fft_coefficient__coeff_39__attr_"abs"']
# , 'Global_active_power__fft_coefficient__coeff_39__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_39__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_39__attr_"real"', 'Global_active_power__fft_coefficient__coeff_3__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_3__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_3__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_3__attr_"real"', 'Global_active_power__fft_coefficient__coeff_40__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_40__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_40__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_40__attr_"real"', 'Global_active_power__fft_coefficient__coeff_41__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_41__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_41__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_41__attr_"real"', 'Global_active_power__fft_coefficient__coeff_42__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_42__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_42__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_42__attr_"real"', 'Global_active_power__fft_coefficient__coeff_43__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_43__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_43__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_43__attr_"real"', 'Global_active_power__fft_coefficient__coeff_44__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_44__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_44__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_44__attr_"real"', 'Global_active_power__fft_coefficient__coeff_45__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_45__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_45__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_45__attr_"real"', 'Global_active_power__fft_coefficient__coeff_46__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_46__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_46__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_46__attr_"real"', 'Global_active_power__fft_coefficient__coeff_47__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_47__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_47__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_47__attr_"real"', 'Global_active_power__fft_coefficient__coeff_48__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_48__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_48__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_48__attr_"real"', 'Global_active_power__fft_coefficient__coeff_49__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_49__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_49__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_49__attr_"real"', 'Global_active_power__fft_coefficient__coeff_4__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_4__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_4__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_4__attr_"real"', 'Global_active_power__fft_coefficient__coeff_50__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_50__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_50__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_50__attr_"real"', 'Global_active_power__fft_coefficient__coeff_51__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_51__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_51__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_51__attr_"real"', 'Global_active_power__fft_coefficient__coeff_52__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_52__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_52__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_52__attr_"real"', 'Global_active_power__fft_coefficient__coeff_53__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_53__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_53__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_53__attr_"real"', 'Global_active_power__fft_coefficient__coeff_54__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_54__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_54__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_54__attr_"real"', 'Global_active_power__fft_coefficient__coeff_55__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_55__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_55__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_55__attr_"real"', 'Global_active_power__fft_coefficient__coeff_56__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_56__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_56__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_56__attr_"real"', 'Global_active_power__fft_coefficient__coeff_57__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_57__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_57__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_57__attr_"real"', 'Global_active_power__fft_coefficient__coeff_58__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_58__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_58__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_58__attr_"real"', 'Global_active_power__fft_coefficient__coeff_59__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_59__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_59__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_59__attr_"real"', 'Global_active_power__fft_coefficient__coeff_5__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_5__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_5__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_5__attr_"real"', 'Global_active_power__fft_coefficient__coeff_60__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_60__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_60__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_60__attr_"real"', 'Global_active_power__fft_coefficient__coeff_61__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_61__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_61__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_61__attr_"real"', 'Global_active_power__fft_coefficient__coeff_62__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_62__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_62__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_62__attr_"real"', 'Global_active_power__fft_coefficient__coeff_63__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_63__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_63__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_63__attr_"real"', 'Global_active_power__fft_coefficient__coeff_64__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_64__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_64__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_64__attr_"real"', 'Global_active_power__fft_coefficient__coeff_65__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_65__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_65__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_65__attr_"real"', 'Global_active_power__fft_coefficient__coeff_66__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_66__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_66__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_66__attr_"real"', 'Global_active_power__fft_coefficient__coeff_67__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_67__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_67__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_67__attr_"real"', 'Global_active_power__fft_coefficient__coeff_68__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_68__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_68__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_68__attr_"real"', 'Global_active_power__fft_coefficient__coeff_69__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_69__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_69__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_69__attr_"real"', 'Global_active_power__fft_coefficient__coeff_6__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_6__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_6__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_6__attr_"real"', 'Global_active_power__fft_coefficient__coeff_70__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_70__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_70__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_70__attr_"real"', 'Global_active_power__fft_coefficient__coeff_71__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_71__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_71__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_71__attr_"real"', 'Global_active_power__fft_coefficient__coeff_72__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_72__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_72__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_72__attr_"real"', 'Global_active_power__fft_coefficient__coeff_73__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_73__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_73__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_73__attr_"real"', 'Global_active_power__fft_coefficient__coeff_74__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_74__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_74__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_74__attr_"real"', 'Global_active_power__fft_coefficient__coeff_75__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_75__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_75__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_75__attr_"real"', 'Global_active_power__fft_coefficient__coeff_76__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_76__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_76__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_76__attr_"real"', 'Global_active_power__fft_coefficient__coeff_77__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_77__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_77__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_77__attr_"real"', 'Global_active_power__fft_coefficient__coeff_78__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_78__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_78__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_78__attr_"real"', 'Global_active_power__fft_coefficient__coeff_79__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_79__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_79__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_79__attr_"real"', 'Global_active_power__fft_coefficient__coeff_7__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_7__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_7__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_7__attr_"real"', 'Global_active_power__fft_coefficient__coeff_80__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_80__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_80__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_80__attr_"real"', 'Global_active_power__fft_coefficient__coeff_81__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_81__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_81__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_81__attr_"real"', 'Global_active_power__fft_coefficient__coeff_82__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_82__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_82__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_82__attr_"real"', 'Global_active_power__fft_coefficient__coeff_83__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_83__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_83__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_83__attr_"real"', 'Global_active_power__fft_coefficient__coeff_84__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_84__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_84__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_84__attr_"real"', 'Global_active_power__fft_coefficient__coeff_85__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_85__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_85__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_85__attr_"real"', 'Global_active_power__fft_coefficient__coeff_86__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_86__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_86__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_86__attr_"real"', 'Global_active_power__fft_coefficient__coeff_87__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_87__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_87__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_87__attr_"real"', 'Global_active_power__fft_coefficient__coeff_88__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_88__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_88__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_88__attr_"real"', 'Global_active_power__fft_coefficient__coeff_89__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_89__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_89__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_89__attr_"real"', 'Global_active_power__fft_coefficient__coeff_8__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_8__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_8__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_8__attr_"real"', 'Global_active_power__fft_coefficient__coeff_90__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_90__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_90__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_90__attr_"real"', 'Global_active_power__fft_coefficient__coeff_91__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_91__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_91__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_91__attr_"real"', 'Global_active_power__fft_coefficient__coeff_92__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_92__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_92__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_92__attr_"real"', 'Global_active_power__fft_coefficient__coeff_93__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_93__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_93__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_93__attr_"real"', 'Global_active_power__fft_coefficient__coeff_94__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_94__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_94__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_94__attr_"real"', 'Global_active_power__fft_coefficient__coeff_95__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_95__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_95__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_95__attr_"real"', 'Global_active_power__fft_coefficient__coeff_96__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_96__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_96__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_96__attr_"real"', 'Global_active_power__fft_coefficient__coeff_97__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_97__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_97__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_97__attr_"real"', 'Global_active_power__fft_coefficient__coeff_98__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_98__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_98__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_98__attr_"real"', 'Global_active_power__fft_coefficient__coeff_99__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_99__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_99__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_99__attr_"real"', 'Global_active_power__fft_coefficient__coeff_9__attr_"abs"', 'Global_active_power__fft_coefficient__coeff_9__attr_"angle"', 'Global_active_power__fft_coefficient__coeff_9__attr_"imag"', 'Global_active_power__fft_coefficient__coeff_9__attr_"real"', 'Global_active_power__first_location_of_maximum', 'Global_active_power__first_location_of_minimum', 'Global_active_power__has_duplicate_max', 'Global_active_power__has_duplicate_min', 'Global_active_power__kurtosis', 'Global_active_power__large_standard_deviation__r_0.05', 'Global_active_power__large_standard_deviation__r_0.1', 'Global_active_power__large_standard_deviation__r_0.15000000000000002', 'Global_active_power__large_standard_deviation__r_0.2', 'Global_active_power__large_standard_deviation__r_0.25', 'Global_active_power__large_standard_deviation__r_0.30000000000000004', 'Global_active_power__last_location_of_maximum', 'Global_active_power__last_location_of_minimum', 'Global_active_power__length', 'Global_active_power__linear_trend__attr_"intercept"', 'Global_active_power__linear_trend__attr_"pvalue"', 'Global_active_power__linear_trend__attr_"rvalue"', 'Global_active_power__linear_trend__attr_"slope"', 'Global_active_power__linear_trend__attr_"stderr"', 'Global_active_power__longest_strike_above_mean', 'Global_active_power__longest_strike_below_mean', 'Global_active_power__maximum', 'Global_active_power__mean', 'Global_active_power__mean_abs_change', 'Global_active_power__mean_change', 'Global_active_power__mean_second_derivative_central', 'Global_active_power__median', 'Global_active_power__minimum', 'Global_active_power__number_crossing_m__m_0', 'Global_active_power__number_crossing_m__m_1', 'Global_active_power__number_cwt_peaks__n_1', 'Global_active_power__number_cwt_peaks__n_5', 'Global_active_power__number_peaks__n_1', 'Global_active_power__number_peaks__n_10', 'Global_active_power__number_peaks__n_3', 'Global_active_power__number_peaks__n_5', 'Global_active_power__number_peaks__n_50', 'Global_active_power__percentage_of_reoccurring_datapoints_to_all_datapoints', 'Global_active_power__percentage_of_reoccurring_values_to_all_values', 'Global_active_power__quantile__q_0.1', 'Global_active_power__quantile__q_0.2', 'Global_active_power__quantile__q_0.3', 'Global_active_power__quantile__q_0.4', 'Global_active_power__quantile__q_0.6', 'Global_active_power__quantile__q_0.7', 'Global_active_power__quantile__q_0.8', 'Global_active_power__quantile__q_0.9', 'Global_active_power__range_count__max_1000000000000.0__min_0', 'Global_active_power__range_count__max_1__min_-1', 'Global_active_power__ratio_beyond_r_sigma__r_0.5', 'Global_active_power__ratio_beyond_r_sigma__r_1', 'Global_active_power__ratio_beyond_r_sigma__r_1.5', 'Global_active_power__ratio_beyond_r_sigma__r_10', 'Global_active_power__ratio_beyond_r_sigma__r_2', 'Global_active_power__ratio_beyond_r_sigma__r_2.5', 'Global_active_power__ratio_beyond_r_sigma__r_3', 'Global_active_power__ratio_beyond_r_sigma__r_5', 'Global_active_power__ratio_beyond_r_sigma__r_6', 'Global_active_power__ratio_beyond_r_sigma__r_7', 'Global_active_power__ratio_value_number_to_time_series_length', 'Global_active_power__sample_entropy', 'Global_active_power__skewness', 'Global_active_power__spkt_welch_density__coeff_2', 'Global_active_power__spkt_welch_density__coeff_5', 'Global_active_power__spkt_welch_density__coeff_8', 'Global_active_power__standard_deviation', 'Global_active_power__sum_of_reoccurring_data_points', 'Global_active_power__sum_of_reoccurring_values', 'Global_active_power__sum_values', 'Global_active_power__symmetry_looking__r_0.05', 'Global_active_power__symmetry_looking__r_0.1', 'Global_active_power__symmetry_looking__r_0.15000000000000002', 'Global_active_power__symmetry_looking__r_0.2', 'Global_active_power__symmetry_looking__r_0.25', 'Global_active_power__symmetry_looking__r_0.30000000000000004', 'Global_active_power__symmetry_looking__r_0.35000000000000003', 'Global_active_power__symmetry_looking__r_0.4', 'Global_active_power__symmetry_looking__r_0.45', 'Global_active_power__symmetry_looking__r_0.5', 'Global_active_power__symmetry_looking__r_0.55', 'Global_active_power__symmetry_looking__r_0.6000000000000001', 'Global_active_power__symmetry_looking__r_0.65'
# , 'Global_active_power__symmetry_looking__r_0.7000000000000001', 'Global_active_power__symmetry_looking__r_0.75', 'Global_active_power__symmetry_looking__r_0.8', 'Global_active_power__symmetry_looking__r_0.8500000000000001', 'Global_active_power__symmetry_looking__r_0.9', 'Global_active_power__symmetry_looking__r_0.9500000000000001', 'Global_active_power__time_reversal_asymmetry_statistic__lag_1', 'Global_active_power__time_reversal_asymmetry_statistic__lag_2', 'Global_active_power__time_reversal_asymmetry_statistic__lag_3', 'Global_active_power__value_count__value_0', 'Global_active_power__value_count__value_1', 'Global_active_power__variance', 'Global_active_power__variance_larger_than_standard_deviation']
# imp_num_cols=['power(t6)','power(t5)','power(t4)','power(t3)','power(t2)','power(t1)','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3','sub_metering_4','average_7_days','max_week_power','min_week_power','Weekend_or_not']
# imp_label=['active_power_t+1']
imp_label = ['active_power_t+1','active_power_t+2','active_power_t+3','active_power_t+4','active_power_t+5','active_power_t+6','active_power_t+7']


# The different features after feature extraction are:

# In[586]:


print((list(Features_db1.columns)))


# In[587]:


data_final1= data_final.replace(np.inf,0)
data_final1=data_final1.replace(-np.inf,0)


# In[588]:


data_final1 = data_final1.astype('float64')


# In[589]:


#dividing into train and test
n_days = int(len(data_final1) * 0.80)
train_data= data_final[:n_days]
test_data= data_final[n_days:]


# We create test and train data based on our declred important columns.

# In[590]:


train_X = train_data[imp_train_cols]
test_X = test_data[imp_train_cols]
train_Y = train_data[imp_label]
test_Y = test_data[imp_label]



# Below is the final data representation with 337 variables which go for training and 7 columns in target label which go for training one by one .

# In[607]:


train_X.head()


# In[608]:


train_Y.head()


# Below We see that the distribution of labels is gaussian so i haven't transformed the label. I could have took log transformation is it was skewed and then predict.

# In[600]:


#checking the distribution of output variable. Looks like Gaussian
import matplotlib.pyplot as plot
bin_values = np.arange(start = 0, stop = 10000,step =10)
train_Y.hist(bins=bin_values,figsize=[14,6])
plot.show()


# Due to lack of time , I couldn't run gridsearch or random search to tune hyperparamters for out xgboost model. But, surely model performance will improve if we tune them. Here I am taking values of paramters with hit and trial method and best to my understanding for now.

# In[593]:


param = {'max_depth':3,'eta':0.02,'silent':1,'subsample':0.6,'reg_lambda':1.5,'reg_alpha':0.001,
        'min_child_weight':7, 'colsample_bytree':0.85, 'nthread':32,'gamma':0.01,'objective':'reg:linear','tree_method':'approx',
        'booster':'gbtree'}


# Prediction: We run models for predictiong each day in future upto 7 days and predict the total Global power consumption.

# In[1]:


yhat_sequence = list()
for i in range(0,7):
    train_Y1= train_Y.values[:,i].ravel()
    train_Y1 = train_Y1.astype('float64')
    test_Y1= test_Y.values[:,i].ravel()
    test_Y1 = test_Y1.astype('float64')
    train_dmatrix=xg.DMatrix(train_X,train_Y1)
    model = xg.train(param,train_dmatrix,num_boost_round = 800)
    test_dmatrix=xg.DMatrix(test_X.iloc[i:i+1,:])
    pred_test = model.predict(test_dmatrix)
    # add to the result
    yhat_sequence.append(pred_test[0])


# The predicted values for next 7 days are stored in yhat_sequence and plotted in graph with (actual vs predicted values) as shown below.

# Also,to reduce the dimensionality and improve performance of our model , we can run xgboost feature_importance to get important features explaining our model and use the top features. I am skipping that part here because of lack of time but its just a simple code and can be implemented to reduce model complexity.

# In[596]:


print(yhat_sequence)


# In[604]:


result =np.array(test_Y.head(1))
actual = actual.flatten() 
print(actual)
#actual.shape


# Below shown is the scatter plot for actual and predicted values where the points close to the 45 degree line represents predictions which are closer to he actual values.

# In[602]:



fig,ax = plt.subplots()
ax.scatter(actual, yhat_sequence)
ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
fig.show()


# The plot comparing predicted and actual values.
# The mean absolute error and root mean squared error are calculated for the next 7 days predicted values and next 7 days original values as shown below. We get RMSE of 276 whereas MAE is 221.

# In[603]:


print(mean_absolute_error(actual,yhat_sequence))
print(math.sqrt(mean_squared_error(actual,yhat_sequence)))

 
