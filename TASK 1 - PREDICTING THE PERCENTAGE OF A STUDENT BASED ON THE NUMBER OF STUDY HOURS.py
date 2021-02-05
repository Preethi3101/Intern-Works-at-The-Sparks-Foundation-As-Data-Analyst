#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
url="http://bit.ly/w-data"
data=pd.read_csv(url)


# In[2]:


data.head()


# In[3]:


data.describe()


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(x="Hours",y="Scores",data=data)


# In[16]:


x=np.array(data["Hours"]).reshape(-1,1)
y=np.array(data["Scores"])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[7]:


from sklearn.linear_model import LinearRegression
score=LinearRegression()
score.fit(x_train,y_train)


# In[8]:


print(score.coef_)


# In[9]:


print(score.intercept_)


# In[10]:


regline=(score.coef_*x)+(score.intercept_)
plt.scatter(x,y)
plt.plot(x,regline)
plt.show()


# In[11]:


y_prediction=score.predict(x_test)
df=pd.DataFrame({"Actual":y_test,"Predicted":y_prediction})
print(df)


# In[12]:


hour=[[7.45]]
p_score=score.predict(hour)
print("If study hour is 7.45, then percentage of mark is",p_score)


# In[13]:


hour=[[9.25]]
p_score=score.predict(hour)
print("If study hour is 9.25, then percentage of mark is",p_score)


# In[14]:


accuracy=score.score(x_test,y_test)
print("The accuracy of applied model is",accuracy*100)


# In[ ]:




