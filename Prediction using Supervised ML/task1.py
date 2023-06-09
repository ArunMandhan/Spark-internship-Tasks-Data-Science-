import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Reading the Data 
data = pd.read_csv ('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
print("Data imported sucessfully")
data.head(10)
# Check if there any null value in the Dataset
data.isnull == True
sns.set_style('darkgrid')
sns.scatterplot(x= data['Hours'],y= data['Scores'])
plt.title('Marks Percentage Vs Study Hours',size=25)
plt.ylabel('Marks Percentage', size=15)
plt.xlabel('Hours Studied', size=15)
plt.show()

sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=25)
plt.ylabel('Marks Percentage', size=15)
plt.xlabel('Hours Studied', size=15)
plt.show()
print(data.corr())

# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

# Spliting the Data in two
train_X, actual_X, train_y, actual_y = train_test_split(X, y, test_size=0.2, random_state = 0)

regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")

pred_y = regression.predict(actual_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in actual_X], 'Predicted Marks': [k for k in pred_y]})
prediction

compare_scores = pd.DataFrame({'Actual Marks': actual_y, 'Predicted Marks': pred_y})
compare_scores

plt.scatter(x=actual_X, y=actual_y, color='blue')
plt.plot(actual_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=25)
plt.ylabel('Marks Percentage', size=15)
plt.xlabel('Hours Studied', size=15)
plt.show()

# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(actual_y,pred_y))

hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))