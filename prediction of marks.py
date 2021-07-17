#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


url="http://bit.ly/w-data"
s_data =pd.read_csv(url)
print("data imported successfully")
s_data.head(10)


# In[17]:


s_data.plot(x="Hours",y="Scores",style="o")
plt.title("Hours vs Percentage")
plt.xlabel("hours studied")
plt.ylabel("percentage scored")
plt.show()


# In[52]:


x=s_data.iloc[:,:-1].values
y=s_data.iloc[:,1].values


# In[53]:


y,x


# In[20]:


x=s_data[['Hours']].values
y=s_data[['Scores']].values


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[50]:


from sklearn.linear_model import LinearRegression
Regressor=LinearRegression()
Regressor.fit(x_train,y_train)
print("training completed")


# In[56]:


line=Regressor.coef_*x+Regressor.intercept_
plt.scatter(x,y)
plt.plot(x,line);
plt.show()


# In[30]:


print(x_test)
y_pred=Regressor.predict(x_test)


# In[61]:


df=pd.DataFrame({'Actual':y_test ,'Predicted':y_pred})
df


# In[49]:


score_pred=np.array([9.25])
score_pred=score_pred.reshape(-1,1)
predict=Regressor.predict(score_pred)
print("No of Hours={}".format(9.25))
print("Predicted score={}".format(predict[0]))


# In[62]:


from sklearn import metrics
print("Mean Absolute Error:" ,metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:




