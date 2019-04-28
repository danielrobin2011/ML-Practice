#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt	


# In[2]:


dataset = pd.read_csv("Salary_Data.csv");


# In[82]:


#Divide the dataset into X and Y

alpha = 0.7
theta = [ -1.0, 1.0 ]

X = dataset.iloc[:,0].values
Y = dataset.iloc[:,1].values
m = len(dataset.iloc[:,0].index)


# In[34]:


# In[8]:


#function for cost function

def cost(X, Y, m, theta):
	squared_sum=0.0
	for i in range(m):
		squared_sum = squared_sum + ((theta[0] + theta[1]*X[i]) - Y[i])**2;
	return ((1.0/(2*m))*squared_sum);


# In[101]:


#Normalize the dataset

X = (X - np.mean(X))/(np.max(X) - np.min(X))
Y = (Y - np.mean(Y))/(np.max(Y) - np.min(Y))

plt.plot(X,Y,'ro')


# In[105]:


def hypothesisSummation(X, Y, theta, isSecondTerm, m):
    sum = 0
    print(theta)
    for i in range(m):
        if(not isSecondTerm):
            sum = sum + ((theta[0] + theta[1]*X[i]) - Y[i])
        else:
            sum = sum + ((theta[0] + theta[1]*X[i]) - Y[i])*X[i];
    return sum;


# In[ ]:


#run Gradient Descent

while(cost(X, Y, m, theta)>0.0001):
    temp0 = theta[0] - alpha*(1/m)*hypothesisSummation(X, Y, theta, False, m);
    temp1 = theta[1] - alpha*(1/m)*hypothesisSummation(X, Y, theta, True, m);
    theta[0] = temp0;
    theta[1] = temp1;
    
print(theta);

# In[ ]:




