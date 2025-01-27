import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load the dataset
data = pd.read_csv("Airline-Dataset-Updated.csv")

# Select relevant columns: 'age' and 'flight status'
data = data[['Age', 'Flight Status']]

# Drop rows with missing values
data.dropna(inplace=True)

# Map 'flight status' to numerical values (e.g., Delayed = 1, On Time = 0)
data['Flight Status'] = data['Flight Status'].map({'Delayed': 1, 'On Time': 0, 'Cancelled': 2})

# Define features (X) and target (y)
X = data[['Age']].values  # Reshape to 2D array
y = data['Flight Status'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=['Age'], class_names=['On Time', 'Delayed', 'Cancelled'], filled=True)
plt.title('Decision Tree for Flight Status Prediction')
plt.show()

# Age group analysis
age_groups = pd.cut(data['Age'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '65+'])
age_group_analysis = data.groupby(age_groups)['Flight Status'].mean()

# Plot age group vs. delay probability
plt.figure(figsize=(10, 6))
age_group_analysis.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel('Age Group')
plt.ylabel('Proportion of Delayed Flights')
plt.title('Proportion of Delayed Flights by Age Group')
plt.xticks(rotation=45)
plt.show()