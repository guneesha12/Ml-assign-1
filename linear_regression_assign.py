#data preprocessing

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
X_train=pd.read_csv('Linear_X_Train.csv')
y_train=pd.read_csv('Linear_Y_Train.csv')
X_test=pd.read_csv('Linear_X_Test.csv')
y_test=y_train.loc[0:1249, :]


#splitting the dataset into training and testing data
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

# feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)'''


# fitting simple linear regression to training set 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# predicting the test set results
y_pred=regressor.predict(X_test)

# visualising the testing data
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Performance vs time(training set)')
plt.xlabel('Time')
plt.ylabel('Performance')
plt.show()

# visualising the testing data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Performance vs time(testing set)')
plt.xlabel('Time')
plt.ylabel('Performance')
plt.show()

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
