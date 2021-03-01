#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import general lib

import pandas as pd
import numpy as np


# In[ ]:





# In[2]:


#Reading training data
data = pd.read_csv(r"C:\Users\Shelvi Garg\Desktop\Code\data-science\JOBATHON 1\TRAIN.csv")


# In[3]:


data.head()


# In[ ]:





# In[4]:


#exploring column names
data.columns


# In[5]:


data.shape


# In[6]:


#processing the data: replacing NaN values with most frequent values in respective columns

for i in range(14):
     (data.iloc[:,i]).fillna(data.iloc[:,i].mode().iloc[0],inplace=True)

print(data)


# In[7]:


data


# In[8]:


#description of our training data 

data.describe()


# In[ ]:





# In[9]:


#assigning target
y = data.Response


# In[10]:


#pre-processing the 2 columns which are combination of str+int,as one-hot decoder cant be applied to these

print(data['Health Indicator'].value_counts())

data["Holding_Policy_Duration"].value_counts()


# In[ ]:





# In[11]:


#pre-processing the 2 columns which are combination of str+int,as one-hot decoder cant be applied to these

data['Health Indicator'].replace({"X1":1,"X2":2,"X3":3,"X4":4,"X5":5,"X6":6,"X7":7,"X8":8,"X9":9},inplace=True)

data['Holding_Policy_Duration'].replace({"14+":"15"},inplace=True)
#need to replace string with float
data['Holding_Policy_Duration'].replace({"15":int("15")},inplace=True)


# In[ ]:





# In[12]:


data = data.reset_index()


# In[13]:


data


# In[ ]:





# #One hot encoding

# In[14]:


# Import label encoder 

from sklearn import preprocessing
# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 

data['City_Code']= label_encoder.fit_transform(data['City_Code'])
data['Accomodation_Type']= label_encoder.fit_transform(data['Accomodation_Type'])
data['Reco_Insurance_Type']= label_encoder.fit_transform(data['Reco_Insurance_Type'])
data['Is_Spouse']= label_encoder.fit_transform(data['Is_Spouse'])



print(data.head())

#Defining Features AS x

x = data[["City_Code","Region_Code",
                 "Accomodation_Type","Reco_Insurance_Type","Upper_Age",
                 "Lower_Age","Is_Spouse","Health Indicator","Holding_Policy_Duration",
                 "Holding_Policy_Type","Reco_Policy_Cat","Reco_Policy_Premium"]]


# In[ ]:





# In[15]:


#Building Model

from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
model = DecisionTreeRegressor(random_state=1)

# Fit model
model.fit(x, y)


# In[16]:


#making predictings on training data

print("Making predictions for the following 5 rows:")
print(x.head())
print("The predictions are")
print(model.predict(x.head()))


# In[17]:


#Making predictions for complete training data

print(model.predict(x))


# In[ ]:





# In[18]:


#calculating the mean error 

from sklearn.metrics import mean_absolute_error

prediction = model.predict(x)
mean_absolute_error(y, prediction)

#result no error : may have overfitted


# In[19]:


#loading testing data

test= pd.read_csv(r"C:\Users\Shelvi Garg\Desktop\Code\data-science\JOBATHON 1\TEST.csv")
test.shape


# In[20]:


test.head()


# In[21]:


#Processing Test data to replace NaN Values

for i in range(12):
     (test.iloc[:,i]).fillna(test.iloc[:,i].mode().iloc[0],inplace=True)

print(test)
test.shape


# In[22]:


#pre-processing the 2 columns which are combination of str+int,as one-hot decoder cant be applied to these
#same as for training data
test['Health Indicator'].replace({"X1":1,"X2":2,"X3":3,"X4":4,"X5":5,"X6":6,"X7":7,"X8":8,"X9":9},inplace=True)

test['Holding_Policy_Duration'].replace({"14+":"15"},inplace=True)
test['Holding_Policy_Duration'].replace({"15":int("15")},inplace=True)


# In[23]:


#One-hot-encoder

# Import label encoder 

from sklearn import preprocessing
# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'Country'. 

test['City_Code']= label_encoder.fit_transform(test['City_Code'])
test['Accomodation_Type']= label_encoder.fit_transform(test['Accomodation_Type'])
test['Reco_Insurance_Type']= label_encoder.fit_transform(test['Reco_Insurance_Type'])
test['Is_Spouse']= label_encoder.fit_transform(test['Is_Spouse'])


# In[24]:


test.head()


# In[25]:


test.shape


# In[26]:


test.columns


# In[27]:


#filtering columns we need to provide to predict

test_data= test[["City_Code","Region_Code",
                 "Accomodation_Type","Reco_Insurance_Type","Upper_Age",
                 "Lower_Age","Is_Spouse","Health Indicator","Holding_Policy_Duration",
                 "Holding_Policy_Type","Reco_Policy_Cat","Reco_Policy_Premium"]]


# In[28]:


#predicting
test_prediction = model.predict(test_data)


# In[29]:


test_prediction


# In[30]:


#length of resultant array;

len(test_prediction)


# In[31]:


#converting result array to pandas dataframe and finally to CSV as to save for submission

pd.DataFrame(test_prediction).to_csv(r"C:\Users\Shelvi Garg\Desktop\Code\data-science\result1.csv")


# In[32]:


result_data = pd.read_csv(r"C:\Users\Shelvi Garg\Desktop\Code\data-science\result1.csv")


# In[33]:


print(result_data)


# In[37]:


result_data.isnull()


# In[ ]:




