
# coding: utf-8

# In[98]:

import pandas as pd
import numpy as np
import sklearn


# In[99]:

Bkr = pd.read_csv("F:\Back order prediction\Training_Dataset_v2.csv")


# In[100]:

Bkr.shape


# In[101]:

bkr = Bkr.sample(n=10000,replace="False")


# In[102]:

bkr.shape


# In[103]:

bkr.isnull().sum() # Null values in lead time is more.


# In[104]:


bkr.lead_time.value_counts().sum()


# In[105]:

bkr.lead_time.isnull().value_counts() #   Lead time : Transit time for product (if available)


# In[106]:

bkr.info()


# In[107]:

bkr.drop(["sku"],axis = 1 , inplace = True)  # Dropping the sku ..


# In[108]:

bkr.describe()


# In[109]:

bkr.shape


# In[110]:


bkr.columns


# In[111]:

## Lead time has Maximum NA Values So converting it to Mean .... Because its time...


# In[112]:

bkr["lead_time"] = bkr["lead_time"].fillna(bkr["lead_time"].mean()) ## Filling nas by its mean because it is time ..


# In[113]:

######### Making categrical variables to numerical ....


# In[114]:

for col in ['potential_issue',
            'deck_risk',
            'oe_constraint',
            'ppap_risk',
            'stop_auto_buy',
            'rev_stop',
            'went_on_backorder']:    
    bkr[col]=pd.factorize(bkr[col])[0]


# In[115]:

bkr.head(5)


# In[116]:

bkr['perf_12_month_avg'] = bkr['perf_12_month_avg'].replace(-99, np.NaN)
bkr['perf_12_month_avg'] = bkr['perf_12_month_avg'].replace(-99, np.NaN)


# In[117]:

#   View count/percentage of missing cells



tot=bkr.isnull().sum().sort_values(ascending=False)
perc=(round(100*bkr.isnull().sum()/bkr.isnull().count(),1)).sort_values(ascending=False)
missing_data = pd.concat([tot, perc], axis=1, keys=['Missing', 'Percent'])
missing_data


# In[118]:

bkr = bkr.fillna(bkr.median(), inplace=True)  #impute the medians


# In[119]:

bkr.isnull().sum()


# In[120]:

### Checking the target variable.


# In[121]:

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8


# In[122]:

# target variable where data is highly giving importance to one category {Product actually went on backorder } 


# In[123]:

sns.countplot(x = "went_on_backorder" , data = bkr ) # Maximum amount of product are not went on backorder.


# In[124]:

### It is observed that the data is imbalanced as it gives importance to onlly one variable ...


# In[125]:

bkr.went_on_backorder.value_counts()


# # Undersampling : To balance the data

# In[126]:

bkr.info()


# In[127]:

#Create independent and Dependent Features
columns = bkr.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["went_on_backorder"]]
# Store the variable we are predicting 
target = "went_on_backorder"
# Define a random state 
state = np.random.RandomState(42)
X = bkr[columns]
Y = bkr[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)


# In[128]:

bkr.isnull().values.any()


# In[129]:

count_classes = pd.value_counts(bkr['went_on_backorder'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Back order distribution")

plt.xticks(range(2))

plt.xlabel("went_on_backorder")

plt.ylabel("Frequency")


# In[130]:

from imblearn.under_sampling import NearMiss


# In[131]:

# Implementing Undersampling for Handling Imbalanced 
nm = NearMiss(sampling_strategy='auto')
X_res,y_res=nm.fit_sample(X,Y)


# In[132]:

X_res.shape,y_res.shape


# In[133]:

from collections import Counter
print('Original dataset shape {}'.format(Counter(Y)))  
print('Resampled dataset shape {}'.format(Counter(y_res)))

Here data get balanced ..Now proceed for Model building

# # Making data standardization :

# In[134]:

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['lead_time', 'sales_1_month', 'min_bank', 'local_bo_qty']
bkr[columns_to_scale] = standardScaler.fit_transform(bkr[columns_to_scale])


# In[135]:

bkr.head()


# In[136]:

bkr.shape


# In[137]:

X=bkr.iloc[:,0:21]


# In[138]:

X.columns


# In[139]:

Y=bkr.iloc[:,-1]


# In[140]:

from sklearn.neighbors import KNeighborsClassifier


# In[141]:

from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,20):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,Y,cv=5)
    knn_scores.append(score.mean())


# In[142]:

print(knn_scores)


# In[143]:

#knn_scores=pd.Series([knn_scores])


# In[144]:

plt.plot([k for k in range(1, 20)], knn_scores, color = 'red')
for i in range(1,20):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 20)])
#knn_scores.plt(figsize=(20,20))
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')

# It Is observed that from 8 onwards it almost gives constant stable line so taking 8 nearest variables.
# In[145]:

knn_classifier = KNeighborsClassifier(n_neighbors = 8)
score=cross_val_score(knn_classifier,X,Y,cv=10)


# In[146]:

score.mean()  # Gives 99 % Accuracy


# In[ ]:




# # Feature Scaling :

# In[48]:

corr=bkr.corr()
fig=plt.figure(figsize=(15,9))

ax=fig.add_subplot(111)
cax=ax.matshow(corr,cmap='coolwarm',vmin=-1,vmax=1)
fig.colorbar(cax)

ticks=np.arange(0,len(bkr.columns),1)
ax.set_xticks(ticks)

plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(bkr.columns)
ax.set_yticklabels(bkr.columns)

plt.show()

From above Corr Plt observed that their are some variables like Forecast 3,6 and 9 months and sales 1,3,6 and 9 are highly corelated. So Taking only one of them.
# In[147]:

bkr = bkr.drop(['forecast_6_month', 'forecast_9_month' ,'sales_3_month' ,'sales_6_month' , 'sales_9_month' ], axis=1)


# In[148]:

bkr.columns


# In[149]:

bkr.shape

# Now taking only these 17 variables ..
# In[150]:

x = bkr.iloc[:,0:16]


# In[151]:

x.columns


# In[152]:

y = bkr.iloc[:,-1]


# In[56]:

X_train, X_test, y_train, y_test = train_test_split( 
			x, y, test_size = 0.3) 


# In[57]:

X_train.shape


# In[58]:

y_train.shape


# In[59]:

X_test.shape


# In[60]:

y_test.shape


# In[ ]:




# # Logistic regression :

# In[61]:

from sklearn.linear_model import LogisticRegression

#lireg=LinearRegression().fit(x1_train,y1_train) # initialize & fit the model
#y_pred=lireg.predict(x1_test)


# In[62]:

logmodel = LogisticRegression().fit(X_train,y_train)


# In[63]:

predict = logmodel.predict(X_test)


# In[64]:

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score


# In[65]:

print(confusion_matrix(y_test,predict))


# In[66]:

print(classification_report(y_test,predict))


# # RF :

# In[67]:

from sklearn.ensemble import RandomForestClassifier


# In[68]:

randomforest_classifier= RandomForestClassifier(n_estimators=8)

score=cross_val_score(randomforest_classifier,X,Y,cv=10)


# In[69]:

score.mean()


# In[71]:

from sklearn import metrics


# In[75]:

lm_MAE=metrics.mean_absolute_error(y_test,predict)            ##shows the mean error term values.
print(lm_MAE.round(4))


# In[78]:

lm_MSE=metrics.mean_squared_error(y_test,predict)           ##  mean squre erroe value 
print(lm_MSE.round(2))


# In[81]:

lcn_RMSE=pow(lm_MSE,.5)                                         ## RMSE value lower the value better the value..
print(lcn_RMSE.round(2))


# In[85]:

from sklearn.metrics import r2_score                           ### r2 values negative...
r2_score(y_test,predict)
#print(r2_score)


# In[ ]:




# In[ ]:




# # DT :

# In[86]:

from sklearn.tree import DecisionTreeClassifier


# In[87]:

dtree=DecisionTreeClassifier(min_samples_split=100)


# In[88]:

dtree.fit(X_train,y_train)


# In[89]:

pred_value=dtree.predict(X_test)


# In[90]:

print(confusion_matrix(y_test,predict))


# In[91]:

print(classification_report(y_test,predict))


# In[ ]:




# In[57]:

Tpr = 14896 /14896 + 16  ### Recall value is important.. As your dataset is imbalanced...
Tpr


# In[58]:

FPR = 88 / 14896 + 16   ### 
FPR


# In[59]:

TNR = 0 


# In[61]:

FNR = 16 / 88


# In[ ]:

##


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



