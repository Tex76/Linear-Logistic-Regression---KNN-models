import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# load data from csv file using pandas
df = pd.read_csv('salary.csv')


# extract features (X) and target variable (y)
X = df[['Level ']]
y = df['Salary']

# split the dataset into training and testing subsets usnig train_test_split method
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# build and fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# provide a summary of the model
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)


# calculate the R-squared score for training subset
print("R-squared score (Training):", model.score(X_train, y_train))

# use the test subset to make predictions
y_pred = model.predict(X_test)

# calculate the R-squared score for test subset
print("R-squared score (Testing):", r2_score(y_test, y_pred))
