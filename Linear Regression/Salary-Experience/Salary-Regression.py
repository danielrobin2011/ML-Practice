import numpy as np;
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv");

#Divide the dataset into X and Y

alpha = 1

X = dataset.iloc[:,0].values
Y = dataset.iloc[:,1].values
m = dataset.iloc[:,0].values.length();
print(m)

theta=[ 1.0, -1.0 ]

def cost(X, Y, m, theta):
	squared_sum=0.0
	for i in range(m):
		squared_sum = squared_sum + ((theta[0] + theta[1]*X[i]) - Y[i])**2;
	return ((1.0/(2*m))*squared_sum);

print(X)
print(Y)

X = (X - np.mean(X))/(np.max(X) - np.min(X))
print(X)

Y = (Y - np.mean(Y))/(np.max(Y) - np.min(Y))
print(Y)

print(cost(X,Y,10,theta));

#while(cost(X,Y,10,theta)<0.00)
#	temp0 = theta[0] - alpha*();
#	np.max

