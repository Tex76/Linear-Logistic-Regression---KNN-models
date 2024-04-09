import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load csv file using pandas
data = pd.read_csv("iris.csv", header=None, names=[
                   'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Mapping of species to numeric values in order to work with KNN
species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data['species'] = data['species'].map(species_map)

# Divide data into features and target
X = data.drop('species', axis=1)
y = data['species']

# Split the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Using MinMaxScaler to scale the features
scaler = MinMaxScaler(feature_range=(1, 3))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train_scaled, y_train)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Predictions
log_reg_train_predictions = log_reg_model.predict(X_train_scaled)
log_reg_test_predictions = log_reg_model.predict(X_test_scaled)

knn_train_predictions = knn_model.predict(X_train_scaled)
knn_test_predictions = knn_model.predict(X_test_scaled)


# Evaluation Metrics
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1


log_reg_train_metrics = evaluate_model(y_train, log_reg_train_predictions)
log_reg_test_metrics = evaluate_model(y_test, log_reg_test_predictions)

knn_train_metrics = evaluate_model(y_train, knn_train_predictions)
knn_test_metrics = evaluate_model(y_test, knn_test_predictions)

# Summary
print("Logistic Regression Model:")
print("Training Set:")
print("Recall:", log_reg_train_metrics[2])
print("Accuracy:", log_reg_train_metrics[0])
print("Precision:", log_reg_train_metrics[1])
print("F1-score:", log_reg_train_metrics[3])
print("\nTesting Set:")
print("Recall:", log_reg_test_metrics[2])
print("Accuracy:", log_reg_test_metrics[0])
print("Precision:", log_reg_test_metrics[1])
print("F1-score:", log_reg_test_metrics[3])

print("\nKNN Model:")
print("Training Set Metrics:")
print("Recall:", knn_train_metrics[2])
print("Accuracy:", knn_train_metrics[0])
print("Precision:", knn_train_metrics[1])
print("F1-score:", knn_train_metrics[3])
print("\nTesting Set Metrics:")
print("Recall:", knn_test_metrics[2])
print("Accuracy:", knn_test_metrics[0])
print("Precision:", knn_test_metrics[1])
print("F1-score:", knn_test_metrics[3])
