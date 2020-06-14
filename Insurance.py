
# # Predicting Insurance Premiums
# The purpose of the Exploratory Data Analysis ,Data Visualization and Machine Learning to use this information to predict charges for the new customers. Our simple dataset contains a few attributes for each person such as Age, Sex, BMI, Children, Smoker, Region and their charges.
# 
# 

# ### Importing necessary Libraries for EDA and Machine Learning 

# In[1]:


import pandas as pd  # Import pandas
import numpy as np   # Import numpy
import seaborn as sns # Import seaborn
import plotly.express as px  # Import plotly express for Interactive Chart 
import matplotlib.pyplot as plt  # Import matplotlib 
import seaborn as seabornInstance
import cufflinks as cf
cf.go_offline()
from matplotlib.animation import FuncAnimation #Import Animation Function
from sklearn.svm import SVR  # Import SVR model
from sklearn.model_selection import train_test_split # train test split
from sklearn.linear_model import LinearRegression #Import Linear Regression model
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from sklearn.tree import DecisionTreeRegressor  # Import Decision Tree Regression model
from sklearn import metrics
from sklearn.model_selection import cross_val_predict  # For K-Fold Cross Validation
from sklearn.metrics import mean_squared_error  # For MSE
from math import sqrt  # For squareroot operation
from sklearn.preprocessing import PolynomialFeatures# Prediction with training dataset:
from plotly.offline import iplot, init_notebook_mode # Standard plotly imports
init_notebook_mode()
init_notebook_mode(connected=True)

get_ipython().run_line_magic('matplotlib', 'inline')


# # Exploratory Data Analysis

# ### Previewing  dataset :

# In[2]:


myData = pd.read_csv('insurance.csv') # read the dataset


# In[3]:


myData.head() # Used .head() to get the first five raws of the dataset


# In[4]:


myData.dtypes # Identification of data types


# There are 7 columns and 1338 raws in the dataset . 

# In[5]:


myData.shape #Used .shape to find the size of the dataset


# # Statistical Summary of Numeric Variables
# Used '.describe()'method below to get statistical summary of numeric variables :
# The minimum age has insurance is 18 and maxiumum age is 64 . The minimum bmi is 15 (severely underweight) and the maximum bmi is 53 (extremely obese).The maximum children are five children for people who has life insurance and the minimum is zero . The minimum life insurance charges is 1121.87 and the  maxiumum charges is 63770.42. 

# In[6]:


myData.describe() # Description of variables 


# In[7]:


myData.info() # get info - there are 7 columns and 1337 entries


# ### Finding null values :
# Used .isnull () method below to check if there are none missing value in the dataset. There is no null value below . 

# In[8]:


myData.isnull().sum() # none missing value 


# # Data Visualization

# ## Customers who have Children by Charges :
# 
# * The first figure below shows that most customers does not have children and very few customers have 4 or 5 children.
# * The second figure below shows that 42.9 % of customers does not have children, 24.2% of customers have one child, 17.9 % have two children, 11.7 % have three children ,1.9 % have four children and 1.3 % have five children. 
# * The third figure below shows that customers who have three children charges more than other and customers who have children five charges less than others. 

# In[9]:


# Simple Histogram
sns.countplot(x="children", data=myData ,palette = 'YlGnBu_r'); # children vs non children
plt.show() # plot the histogram 

# Pie Chart 
labels=myData.children.value_counts().index # children count percentage 
colors=["green","blue","hotpink","yellow","navy","#9b59b6"] # color of pie chart 
sizes=myData.children.value_counts().values
plt.figure(figsize=(7,7)) #plot the figure
plt.pie(sizes,labels=labels,colors=colors,autopct="%1.1f%%")
plt.title("Children Count Precentage",color="saddlebrown",fontsize=15) #title of pie chart


plt.figure(figsize=(15,8)) #figure size
ax = sns.barplot(x='children', y='charges', data=myData)# simple barplot to compare customers who has children by charges
ax.set_title('Children by Charges') #title for barplot 


# ## How does Regions affect Life Insurance Charges ?
# 
# * The first figure shows that more customers are from southeast region and less in northeast region.
# *  The second figure shows that 24.3 % customers from southwest, 27.2% customers from southeast, 24.3% customers from northwest and 24.2% customers from northeast. 
# *  The third figure show that customers from southeast charges more than other regions and customers charges from southwest charges less than other regions. 

# In[10]:


# simple interactive Histogram 
fig = px.histogram(myData, x="region") # compare region who has most / least Life Insurance
fig.show() # plot the interactive histogram for regions 

# Pie Chart 
labels=myData.region.value_counts().index # Compare the regions by percentage 
colors=["green","blue","hotpink","yellow","navy","#9b59b6"] # color of pie chart 
sizes=myData.region.value_counts().values
plt.figure(figsize=(7,7)) #plot the figure
plt.pie(sizes,labels=labels,colors=colors,autopct="%1.1f%%")
plt.title("Regions Count Precentage",color="saddlebrown",fontsize=15) #title of pie chart

plt.figure(figsize=(15,8)) #figure size
ax = sns.barplot(x='region', y='charges', data=myData)# simple barplot to compare regions by charges
ax.set_title('Region by Charges') #title for barplot 


# ## How does smoking affect Life Insurance Charges ?
# 
# * The first figure below shows that Non-smokers’ customers are more compared to smokers’ customers. 
# * The second figure below shows that Insurance charges were high for smokers than non-smokers.
# * The third figure below shows that 79.5 % of customers are non-smokers and 20.5% of customers are smokers.
# * The fourth figure below shows that customers with five children and smokers charges less than other customers. Also it shows that smokers customers who doesn’t have children charges more than other customers. 
# 
# 

# In[11]:


vc = myData["smoker"].value_counts() # count smoker 
print(vc)

# Simple Histogram for Smoker and Non smoker 
sns.countplot(x="smoker", data=myData ,palette = 'YlGnBu_r'); # smoker vs non smoker
plt.show()

# boxplot smoker and non smoker charges 
sns.boxplot(x="smoker", y="charges", data=myData,palette = 'YlGnBu_r');  # smokers insurance costs higher than non smokers
plt.show() # plot the histogram 

# Pie Chart Smoker vs None Smoker precentage
labels=myData.smoker.value_counts().index #labels of the pie chart 
sizes=myData.smoker.value_counts().values # sizes of the pie chart
plt.figure(figsize=(7,7)) # size of Pie Chart
plt.pie(sizes,labels=labels,colors=colors,autopct="%1.1f%%")
plt.title("Smoker vs Non smoker",color="saddlebrown",fontsize=15) # title for pie chart 

#Catplot combined people who smoke and don't smoke and people who has children and doesn't has children 
pal = ["#FA5858", "#58D3F7"] # palette for catplot
sns.catplot(x="children", y="charges", hue="smoker",kind="violin", data=myData, palette = pal)


# ### How age affects life insurance charges ?

# * The first figure below shows that ages between 18 and 19 are the highest number of people who having Life Insurance and age between 64 - 65 are the lowest number of people who having life insurance. 
# * The second figure below shows that the younger customers charges less than older customers . 

# In[12]:


# Plotting the age in the dataset using histogram 
fig = px.histogram(myData, x='age')
fig.show()

plt.figure(figsize=(15,8)) #figure size
ax = sns.barplot(x='age', y='charges', data=myData)# simple barplot to compare regions by charges
ax.set_title('Barplot of age by charges') #title 


# ### How does gender affect life insurance charges ?
# 
# * The first figure below shows that males who smoke are greater than females.
# * The second figure below shows that there are 50.5% of customers are male and 49.5% of customers are female.
# 

# In[13]:


sns.set_style('whitegrid')  # using sns.countplot between sex and smoker female smokers is less than male smokers 
# male none-smokers is less than female non smoker
sns.countplot(x='sex', hue='smoker', data=myData, palette='cubehelix')


labels=myData.sex.value_counts().index 
sizes=myData.sex.value_counts().values 
colors=["cyan","orange","hotpink","green","navy","#9b59b6"]
plt.figure(figsize=(7,7))
plt.pie(sizes,labels=labels,colors=colors,autopct="%1.1f%%")
plt.title("Male vs Female ",color="saddlebrown",fontsize=15)


# * The Interactive bar plot below shows that males charges more than female . 

# In[14]:


myData.iplot(kind='bar', x=['sex'],y='charges') # Interactive Bar Plot


# The interactive 3D plot shows the dark blue is where life insurance charges getting higher and the dark orange is where life insuance cost getting lower by gender.

# In[15]:


myData['sex'] = myData['sex'].replace({"male":1,"female":0}) # Convert the categorical data "smoker" to numeric value
myData2 = myData[["sex","charges","sex"]] # Interactive 3D Plot
data = myData2.iplot(kind='surface', colorscale='rdylbu')


# 
# * The first figure below shows that there is a peak at BMI of 30. More number of people are in the range of BMI between 25-35.
# * The second figure below shows that Insurance charges are high for customers whose BMI is high.

# In[16]:


plt.figure(figsize=(9, 8)) # histograms to plot bmi
sns.distplot(myData['bmi'], color='orange', bins=100, hist_kws={'alpha': 0.4});


# In[17]:


ax = sns.scatterplot(data=myData,x='bmi',y='charges')
ax.set_title("BMI vs Charges")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.show(ax)


# ### Frequency of our data variables 

# By the below histograms we get the max percentage of people in each variable range. We can see that that frequency of people are more in age group ranging from 18-20. Majority of people have body mass index between 30-35. And majority of people have no children who have subscribed for Insurance policy and the max Insurance charges charged were in the range is from 1000-15000 .

# In[18]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
myData.plot(kind="hist", y="age", bins=70, color="b", ax=axes[0][0])
myData.plot(kind="hist", y="bmi", bins=100, color="r", ax=axes[0][1])
myData.plot(kind="hist", y="children", bins=6, color="g", ax=axes[1][0])
myData.plot(kind="hist", y="charges", bins=100, color="orange", ax=axes[1][1])
plt.show()


#  # Correlation
#  
#  There are two strongly correlated values which are charges and smokers.

# In[19]:


num_corr = myData.corr()['charges'] 
features_list = num_corr[abs(num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with Charges:\n{}".format(len(features_list), features_list)) # smoker has strongly correleted values with charges


# Since there is a strong correlation between smoker and charges lets get more insights :

# ### Age vs Charges by Smoker
# The below scatterplot shows the relationship of Age vs Expenses by smoker.There is a correlation between smokers and charges, age and charges. As the age hikes up and smoking ratio increases the insurance premium charges hiked up.

# In[20]:


ax = sns.scatterplot(data=myData,x='age',y='charges',hue='smoker')
ax.set_title("Age vs Charges by Smoker")
plt.xlabel("Smoker (Yes - 1, No - 0)")
plt.ylabel("Charges")
plt.show(ax)


# ### Children vs Charges by Smoker

# Below scatterplot shows that customers who smoke and do not have children charged more than other. Although customers who have smoke and have five children did not charge like other smokers.

# In[21]:


ax = sns.scatterplot(data=myData,x='children',y='charges',hue='smoker')
ax.set_title("Children vs Charges by Smoker")
plt.xlabel("Smoker (Yes - 1, No - 0)")
plt.ylabel("Charges")
plt.show(ax)


# ### BMI vs Charges by Smoker

# The scatterplot below shows that non smokers with high BMI charges less that customers who are smokers with high BMI. 

# In[22]:


ax = sns.scatterplot(data=myData,x='bmi',y='charges',hue='smoker')
ax.set_title("BMI vs Charges by Smoker")
plt.xlabel("Smoker (Yes - 1, No - 0)")
plt.ylabel("Charges")
plt.show(ax)


# ### Checking correlation between all variables with Charges 
# The below pair plots shows the correlation between different variables and their charges.we can see that there is clear corrrelation between the different criteria versus charges.Age,sex,bmi,children ,smoker are independant variable and charges are dependent variable

# In[23]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5)) # Scatter Plot
myData.plot(kind='scatter', x='age', y='charges', alpha=0.5, color='green', ax=axes[0], title="Age vs. Charges")
myData.plot(kind='scatter', x='bmi', y='charges', alpha=0.5, color='red', ax=axes[1], title="bmi vs. Charges")
myData.plot(kind='scatter', x='children', y='charges', alpha=0.5, color='blue', ax=axes[2], title="Children vs. Charges")
plt.show()


# ### Correlogram of Insurance Charges

# The darker color in the figure below shows that how strongly correleted variables and the lighter color shows weak correlations between variables . 

# In[24]:


plt.figure(figsize=(12,10), dpi= 80) 
sns.heatmap(myData.corr(), xticklabels=myData.corr().columns, yticklabels=myData.corr().columns, cmap='RdYlGn', center=0, annot=True) #HeatMap

plt.title('Correlogram of Insurance charges', fontsize=22) # plot the title with font size 22
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show() # Plot the figure


# # Machine Learning Models :

# ## Preparing Data for Machine Learning Algorithms

# In[25]:


myData.head()


# In[26]:


myData = myData.drop("region", axis = 1) 
myData.head()


# In[27]:


# Changing binary categories to 1s and 0s
myData['sex'] = myData['sex'].map(lambda s :1  if s == 'female' else 0)
myData['smoker'] = myData['smoker'].map(lambda s :1  if s == 'yes' else 0)

myData.head()


# In[28]:


X = myData.drop(['charges'], axis = 1)
y = myData.charges


# ## Modeling our data

# Score is the R2 score, which varies between 0 and 100%. It is closely related to the MSE but not the same."r2 is the proportion of the variance in the dependent variable that is predictable from the independent variable(s)".~ Wikipedia . Another definition is “(total variance explained by model) / total variance.” So if it is 100%, the two variables are perfectly correlated, i.e., with no variance at all. A low value would show a low level of correlation, meaning a regression model that is not valid, but not in all cases.

# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
lr = LinearRegression().fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print(lr.score(X_test, y_test))


# In[30]:


results = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
results


# In[31]:


# Normalize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[32]:


pd.DataFrame(X_train).head()


# In[33]:


pd.DataFrame(y_train).head()


# ## Multiple Variables Linear Regression 
# 

# Used multiple variables linear regression :
# Regression Analysis is predictive modelling technique and it estimates the relationship between dependent (target) and independent variable (predictor)
# charges = m1*age + m2*sex + m3*bmi + m4*children + m5*smoker + intercept
# 

# In[34]:


X = myData[['age','sex','bmi','children','smoker']].values  # independent variables
y = myData['charges'].values # dependent variable


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # split the data frame into train and test 


# In[36]:


multiple_linear_reg = LinearRegression(fit_intercept=False)  # Create a instance for Linear Regression model
multiple_linear_reg.fit(X_train, y_train)  # Fit data to the model


# ##  Evaluating Multiple Linear Regression Model

# * Training Accuracy for Multiple Linear Regression Model:  0.70
# * Testing Accuracy for Multiple Linear Regression Model:  0.76

# In[37]:


from sklearn.metrics import r2_score  # For find accuracy with R2 Score

# Prediction with training dataset:
y_pred_MLR_train = multiple_linear_reg.predict(X_train)

# Prediction with testing dataset:
y_pred_MLR_test = multiple_linear_reg.predict(X_test)

# Find training accuracy for this model:
accuracy_MLR_train = r2_score(y_train, y_pred_MLR_train)
print("Training Accuracy for Multiple Linear Regression Model: ", accuracy_MLR_train)

# Find testing accuracy for this model:
accuracy_MLR_test = r2_score(y_test, y_pred_MLR_test)
print("Testing Accuracy for Multiple Linear Regression Model: ", accuracy_MLR_test)

# Find RMSE for training data:
RMSE_MLR_train = sqrt(mean_squared_error(y_train, y_pred_MLR_train))
print("RMSE for Training Data: ", RMSE_MLR_train)

# Find RMSE for testing data:
RMSE_MLR_test = sqrt(mean_squared_error(y_test, y_pred_MLR_test))
print("RMSE for Testing Data: ", RMSE_MLR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_MLR = cross_val_predict(multiple_linear_reg, X, y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_MLR = r2_score(y, y_pred_cv_MLR)
print("Accuracy for 10-Fold Cross Predicted Multiple Linaer Regression Model: ", accuracy_cv_MLR)


# ## Polynomial Regression Model

# In[38]:


polynomial_features = PolynomialFeatures(degree=3)  # Create a PolynomialFeatures instance in degree 3
x_train_poly = polynomial_features.fit_transform(X_train)  # Fit and transform the training data to polynomial
x_test_poly = polynomial_features.fit_transform(X_test)  # Fit and transform the testing data to polynomial

polynomial_reg = LinearRegression(fit_intercept=False)  # Create a instance for Linear Regression model
polynomial_reg.fit(x_train_poly, y_train)  # Fit data to the model


# # Evaluating Polynomial Regression Model
# 
# * Training Accuracy for Polynomial Regression Model:  0.83
# * Testing Accuracy for Polynomial Regression Model:  0.87

# In[39]:


# Prediction with training dataset:
y_pred_PR_train = polynomial_reg.predict(x_train_poly)

# Prediction with testing dataset:
y_pred_PR_test = polynomial_reg.predict(x_test_poly)

# Find training accuracy for this model:
accuracy_PR_train = r2_score(y_train, y_pred_PR_train)
print("Training Accuracy for Polynomial Regression Model: ", accuracy_PR_train)

# Find testing accuracy for this model:
accuracy_PR_test = r2_score(y_test, y_pred_PR_test)
print("Testing Accuracy for Polynomial Regression Model: ", accuracy_PR_test)

# Find RMSE for training data:
RMSE_PR_train = sqrt(mean_squared_error(y_train, y_pred_PR_train))
print("RMSE for Training Data: ", RMSE_PR_train)

# Find RMSE for testing data:
RMSE_PR_test = sqrt(mean_squared_error(y_test, y_pred_PR_test))
print("RMSE for Testing Data: ", RMSE_PR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_PR = cross_val_predict(polynomial_reg, polynomial_features.fit_transform(X), y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_PR = r2_score(y, y_pred_cv_PR)
print("Accuracy for 10-Fold Cross Predicted Polynomial Regression Model: ", accuracy_cv_PR)


# ## Decision Tree Regression Model
# 

# In[40]:


decision_tree_reg = DecisionTreeRegressor(max_depth=5, random_state=13)  # Create a instance for Decision Tree Regression model
decision_tree_reg.fit(X_train, y_train)  # Fit data to the model


# ## Evaluating Decision Tree Regression Model
# 
# * Training Accuracy for Decision Tree Regression Model:  0.87
# * Testing Accuracy for Decision Tree Regression Model:  0.83

# In[41]:


# Prediction with training dataset:
y_pred_DTR_train = decision_tree_reg.predict(X_train)

# Prediction with testing dataset:
y_pred_DTR_test = decision_tree_reg.predict(X_test)

# Find training accuracy for this model:
accuracy_DTR_train = r2_score(y_train, y_pred_DTR_train)
print("Training Accuracy for Decision Tree Regression Model: ", accuracy_DTR_train)

# Find testing accuracy for this model:
accuracy_DTR_test = r2_score(y_test, y_pred_DTR_test)
print("Testing Accuracy for Decision Tree Regression Model: ", accuracy_DTR_test)

# Find RMSE for training data:
RMSE_DTR_train = sqrt(mean_squared_error(y_train, y_pred_DTR_train))
print("RMSE for Training Data: ", RMSE_DTR_train)

# Find RMSE for testing data:
RMSE_DTR_test = sqrt(mean_squared_error(y_test, y_pred_DTR_test))
print("RMSE for Testing Data: ", RMSE_DTR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_DTR = cross_val_predict(decision_tree_reg, X, y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_DTR = r2_score(y, y_pred_cv_DTR)


# ## Random Forest Regression Model
# 

# In[42]:


from sklearn.ensemble import RandomForestRegressor  # Import Random Forest Regression model

random_forest_reg = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=13)  # Create a instance for Random Forest Regression model
random_forest_reg.fit(X_train, y_train)  # Fit data to the model


# NOTE: n_estimators represents the number of trees in the forest. Usually the higher the number of trees the better to learn the data. However, adding a lot of trees can slow down the training process considerably, therefore we do a parameter search to find the sweet spot.

# ## Evaluating Random Forest Regression Model
# 
# * Training Accuracy for Random Forest Regression Model:  0.88
# * Testing Accuracy for Random Forest Regression Model:  0.89

# In[43]:


# Prediction with training dataset:
y_pred_RFR_train = random_forest_reg.predict(X_train)

# Prediction with testing dataset:
y_pred_RFR_test = random_forest_reg.predict(X_test)

# Find training accuracy for this model:
accuracy_RFR_train = r2_score(y_train, y_pred_RFR_train)
print("Training Accuracy for Random Forest Regression Model: ", accuracy_RFR_train)

# Find testing accuracy for this model:
accuracy_RFR_test = r2_score(y_test, y_pred_RFR_test)
print("Testing Accuracy for Random Forest Regression Model: ", accuracy_RFR_test)

# Find RMSE for training data:
RMSE_RFR_train = sqrt(mean_squared_error(y_train, y_pred_RFR_train))
print("RMSE for Training Data: ", RMSE_RFR_train)

# Find RMSE for testing data:
RMSE_RFR_test = sqrt(mean_squared_error(y_test, y_pred_RFR_test))
print("RMSE for Testing Data: ", RMSE_RFR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_RFR = cross_val_predict(random_forest_reg, X, y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_RFR = r2_score(y, y_pred_cv_RFR)
print("Accuracy for 10-Fold Cross Predicted Random Forest Regression Model: ", accuracy_cv_RFR)


# ## Support Vector Machine Model 

# In[44]:


support_vector_reg = SVR(gamma="auto", kernel="linear", C=1000)  # Create a instance for Support Vector Regression model
support_vector_reg.fit(X_train, y_train)  # Fit data to the model


# ## Evaluating Support Vector Machine Model
# 
# * Training Accuracy for Support Vector Regression Model:  0.62
# * Testing Accuracy for Support Vector Regression Model:  0.67

# In[45]:


# Prediction with training dataset:
y_pred_SVR_train = support_vector_reg.predict(X_train)

# Prediction with testing dataset:
y_pred_SVR_test = support_vector_reg.predict(X_test)

# Find training accuracy for this model:
accuracy_SVR_train = r2_score(y_train, y_pred_SVR_train)
print("Training Accuracy for Support Vector Regression Model: ", accuracy_SVR_train)

# Find testing accuracy for this model:
accuracy_SVR_test = r2_score(y_test, y_pred_SVR_test)
print("Testing Accuracy for Support Vector Regression Model: ", accuracy_SVR_test)

# Find RMSE for training data:
RMSE_SVR_train = sqrt(mean_squared_error(y_train, y_pred_SVR_train))
print("RMSE for Training Data: ", RMSE_SVR_train)

# Find RMSE for testing data:
RMSE_SVR_test = sqrt(mean_squared_error(y_test, y_pred_SVR_test))
print("RMSE for Testing Data: ", RMSE_SVR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_SVR = cross_val_predict(support_vector_reg, X, y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_SVR = r2_score(y, y_pred_cv_SVR)
print("Accuracy for 10-Fold Cross Predicted Support Vector Regression Model: ", accuracy_cv_SVR)


# ## Compare all results in one table :
# 
# Our best classifier is our Random Forests using 400 estimators and a max_depth of 5

# In[46]:


# Compare all results in one table
training_accuracies = [accuracy_MLR_train, accuracy_PR_train, accuracy_DTR_train, accuracy_RFR_train, accuracy_SVR_train]
testing_accuracies = [accuracy_MLR_test, accuracy_PR_test, accuracy_DTR_test, accuracy_RFR_test, accuracy_SVR_test]
training_RMSE = [RMSE_MLR_train, RMSE_PR_train, RMSE_DTR_train, RMSE_RFR_train, RMSE_SVR_train]
testing_RMSE = [RMSE_MLR_test, RMSE_PR_test, RMSE_DTR_test, RMSE_RFR_test, RMSE_SVR_test]
cv_accuracies = [accuracy_cv_MLR, accuracy_cv_PR, accuracy_cv_DTR, accuracy_cv_RFR, accuracy_cv_SVR]

parameters = ["fit_intercept=False", "fit_intercept=False", "max_depth=5", "n_estimators=400, max_depth=5", "kernel=”linear”, C=1000"]

table_data = {"Parameters": parameters, "Training Accuracy": training_accuracies, "Testing Accuracy": testing_accuracies, 
              "Training RMSE": training_RMSE, "Testing RMSE": testing_RMSE, "10-Fold Score": cv_accuracies}
model_names = ["Multiple Linear Regression", "Polynomial Regression", "Decision Tree Regression", "Random Forest Regression", "Support Vector Regression"]

table_dataframe = pd.DataFrame(data=table_data, index=model_names)
table_dataframe


# ## R^2 (coefficient of determination) regression score function.
# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

# ## Let's test our best regression on some new data

# In[47]:


input_data = {'age': [35],'sex': ['male'],'bmi': [26],'children': [0],'smoker': ['no'],'region': ['southeast']}
input_data = pd.DataFrame(input_data)
input_data


# In[48]:


# Our simple pre-processing 
input_data.drop(["region"], axis=1, inplace=True) 
input_data['sex'] = input_data['sex'].map(lambda s :1  if s == 'female' else 0)
input_data['smoker'] = input_data['smoker'].map(lambda s :1  if s == 'yes' else 0)
input_data


# In[49]:


# Scale our input data  
input_data = sc.transform(input_data)
input_data


# In[50]:


# Reshape our input data in the format required by sklearn models
input_data = input_data.reshape(1, -1)
print(input_data.shape)
input_data


# In[51]:


# Get our predicted insurance rate for our new customer
random_forest_reg.predict(input_data)


# In[52]:


# Note Standard Scaler remembers your inputs so you can use it still here
print(sc.mean_)
print(sc.scale_)


# # Conclusion :

# * Insurance charges were high for smokers than non-smokers.
# 
# * Insurance charges were high for customers who dont have children and less for who have more number of children.
# 
# * Insurance charges trends up as the age increases.
# 
# * More customers are from southeast region and less in northeast region.
# 
# * Insurance charges are high for customers whose BMI is high.
# 
# * Majority of customers have BMI in the range between 30-35.
# 
# * More customers are in the age group ranging from 18-20.
# 
# * Female non smokers are high when compared to male non smokers.
# 
# * Non smokers are more compared to smokers.
# 
# * Males are 1% more in number compared to females.
# 
# * Majority of customers doesn't have children and very few customers have 4 or 5 children.
# 
# * There is a positive correlation with all independant variables and dependent variable.
# 
# * There is a negative correlation with the Independent variable children and Insurance charges as the children increases insurance charges decreases.
