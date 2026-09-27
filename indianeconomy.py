
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
# get_ipython().system('pip install scikit-learn')
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
# get_ipython().system('pip install joblib')
import joblib


# In[2]:


a={"rigon": ["Maharashtra", "Delhi", "Mumbai"], "revenue":[10000,20000,30000]}
print(a)
b=pd.DataFrame(a)
print(b)


# In[3]:


b.groupby("rigon")["revenue"].sum().plot(kind='bar',title="region wise revenue")


# In[4]:


plot.show


# In[5]:


economy=pd.read_csv("Indian_Economy_Sectorwise_Dataset.csv")
print(economy)


# In[6]:


economy.info()


# In[7]:


economy.describe()


# In[8]:


economy.groupby("Year")["GDP_Lakh_Crore"].sum().plot(kind='bar', title="Yearwise GDP",y="GDP",color="Orange")


# In[9]:


economy.groupby("Sector")["GDP_Lakh_Crore"].sum().plot(kind='bar', title="Yearwise GDP",y="GDP",color="Orange")


# In[10]:


l=LabelEncoder()


# In[11]:


economy["new sector"]=l.fit_transform(economy["Sector"])


# In[12]:


economy


# In[13]:


economy.drop(columns=["Sector"],inplace=True)


# In[14]:


economy


# In[15]:


economy["new Quarter"]=l.fit_transform(economy["Quarter"])


# In[16]:


economy


# In[17]:


economy.drop(columns=["Quarter"],inplace=True)


# In[18]:


economy


# In[19]:


economy.info()


# In[20]:


x=economy[["Year"]]
y=economy[["Imports_Crore"]]


# In[21]:


model=LinearRegression()


# In[22]:


model.fit(x,y)


# In[23]:


model.predict([[2027]])


# In[24]:


print("Model R2 Score:", model.score(x,y))


# In[25]:


joblib.dump(model, "linear_regression_model.pkl")
print("Model saved successfully!")

