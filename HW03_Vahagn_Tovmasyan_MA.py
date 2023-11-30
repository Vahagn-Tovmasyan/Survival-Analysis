#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter, ExponentialFitter
from lifelines.utils import k_fold_cross_validation
import seaborn as sns
import warnings


# In[10]:


data_path = 'telco.csv'
raw_data = pd.read_csv(data_path)


# In[9]:


import os
current_directory = os.getcwd()
print("Current Directory:", current_directory)


# In[38]:


def process_data(data):
    data = data.copy()

    data.drop(['ID'], axis=1, inplace=True)

    categorical_columns = ['region', 'retire', 'marital', 'ed', 'gender', 'voice', 'internet', 'custcat','churn', 'forward']
    data = data.copy()
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    data = data.rename(columns={'churn_Yes': 'churn'})

    return data

data = process_data(raw_data)
data.head()


# In[29]:


weibull_model = WeibullAFTFitter()
log_norm_model = LogNormalAFTFitter()
log_logistic_model = LogLogisticAFTFitter()


# In[40]:


from lifelines import WeibullAFTFitter

weibull_model = WeibullAFTFitter()

weibull_model.fit(data, duration_col='tenure', event_col='churn')

weibull_prediction = weibull_model.predict_survival_function(data).T
weibull_prediction_avg = weibull_prediction.mean()

weibull_model.print_summary()


# In[44]:


from lifelines import LogNormalAFTFitter

log_norm_model = LogNormalAFTFitter()

log_norm_model.fit(data, duration_col='tenure', event_col='churn')

log_norm_prediction = log_norm_model.predict_survival_function(data).T
log_norm_prediction_avg = log_norm_prediction.mean()

log_norm_model.print_summary()


# In[45]:


log_logistic_model = LogLogisticAFTFitter()

log_logistic_model.fit(data, duration_col='tenure', event_col='churn')

log_logistic_prediction = log_logistic_model.predict_survival_function(data).T
log_logistic_prediction_avg = log_logistic_prediction.mean()

log_logistic_model.print_summary()


# In[46]:


weibull_model = WeibullAFTFitter()
log_norm_model = LogNormalAFTFitter()
log_logistic_model = LogLogisticAFTFitter()

weibull_model.fit(data, duration_col='tenure', event_col='churn')
log_norm_model.fit(data, duration_col='tenure', event_col='churn')
log_logistic_model.fit(data, duration_col='tenure', event_col='churn')

aic_weibull = weibull_model.AIC_
aic_log_norm = log_norm_model.AIC_
aic_log_logistic = log_logistic_model.AIC_

print("AIC - Weibull:", aic_weibull)
print("AIC - Log-Normal:", aic_log_norm)
print("AIC - Log-Logistic:", aic_log_logistic)

best_model = min([(aic_weibull, 'Weibull'), (aic_log_norm, 'Log-Normal'), (aic_log_logistic, 'Log-Logistic')])

print("Best Model:", best_model[1])


# After doing the AIC, we find out that the best model is Log-Normal.

# In[51]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))

plt.plot(weibull_prediction_avg, label='Weibull (Average)', linestyle='--', color='blue')
plt.plot(log_norm_prediction_avg, label='Log-Normal (Average)', linestyle='--', color='green')
plt.plot(log_logistic_prediction_avg, label='Log-Logistic (Average)', linestyle='--', color='red')

plt.title('Survival Curves Comparison')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid(True)
plt.show()


# In[52]:


significant_columns = ["address", "age", "internet_Yes", "marital_Unmarried", "tenure", "churn", "custcat_E-service", "custcat_Plus service", "custcat_Total service", "voice_Yes"]


# In[53]:


dropped_data = data[significant_columns]
dropped_data


# In[54]:


log_norm = log_norm_model.fit(dropped_data, duration_col='tenure', event_col='churn')
log_norm_prediction = log_norm.predict_survival_function(dropped_data).T
log_norm_prediction_avg = log_norm_prediction.mean()
log_norm.print_summary()


# In[55]:


from lifelines import LogNormalAFTFitter


log_norm_model = LogNormalAFTFitter()

log_norm = log_norm_model.fit(dropped_data, duration_col='tenure', event_col='churn')

log_norm_prediction = log_norm_model.predict_survival_function(dropped_data).T
log_norm_prediction_avg = log_norm_prediction.mean()

log_norm_model.print_summary()


# In[56]:


clv_data = log_norm_prediction.copy()


# In[70]:


margin = 1000
sequence = range(1,len(clv_data.columns)+1)
r = 0.1


# In[71]:


for i in sequence:
    clv_data.loc[:, i] = clv_data.loc[:, i]/((1+r/12)**(sequence[i-1]-1))


# In[72]:


clv_data["CLV"] = margin * clv_data.sum(axis = 1)
clv_data


# In[74]:


for i in sequence:
    clv_data.loc[:, i] = clv_data.loc[:, i] / ((1 + r / 12) ** (i - 1))

clv_data["CLV"] = margin * clv_data.sum(axis=1)
clv_data


# In[75]:


raw_data["CLV"] = clv_data.CLV


# In[76]:


sns.displot( data = raw_data, x = 'CLV', kind = 'kde', hue = 'marital').set(title = 'Customer marital status CLV density')
sns.displot( data = raw_data, x = 'CLV', kind = 'kde', hue = 'ed').set(title = 'Customer education level CLV density')
sns.displot( data = raw_data, x = 'CLV', kind = 'kde', hue = 'retire').set(title = 'Customer retirement CLV density')
sns.displot(data = raw_data, x = 'CLV', kind = 'kde', hue = 'gender').set(title = 'Customer marital status CLV density')
sns.displot(data = raw_data, x = 'CLV', kind = 'kde', hue = 'custcat').set(title = 'Customer service CLV density')


# In[84]:


dropped_data["CLV"] = clv_data.CLV


# In[85]:


retained_customers = dropped_data[dropped_data['churn'] == 0]
retained_clv = retained_customers['CLV'].sum()


# In[86]:


retention_rate = 0.8
cost_per_customer = 5000
retention_cost = len(dropped_data) * retention_rate * cost_per_customer


# In[88]:


annual_budget = retained_clv - retention_cost
annual_budget


# Customer Lifetime Value (CLV) is a vital metric, revealing a customer's total value over their association with a company.
# 
# Accelerated Failure Time (AFT) models predict event times, such as customer churn, and are useful for understanding customer survival.
# 
# In our study, CLV and AFT show a negative correlation—higher CLV corresponds to lower churn risk.
# 
# Regarding coefficients:
# 
# Positive coefficients mean increasing the variable enhances customer lifetime.
# Negative coefficients signify diminishing the variable reduces customer lifetime.
# Coefficient magnitude indicates the variable's impact, sensitive to data scale.
# "The most valuable segments" include retirees and those without a high school diploma, potentially linked to higher demands among educated or younger individuals.
# 
# Retention suggestions:
# 
# Actively listen to customer feedback, especially from younger customers.
# Offer exclusive perks to loyal customers, particularly targeting younger demographics.
# The company faces challenges with internet-related CLV concerns among customers with internet services.

# In[ ]:




