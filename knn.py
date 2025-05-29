import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load data
data = pd.read_csv('dataset.csv')

# Check the first few rows and shape of the data
print(data.head())
print(data.shape)

# Separate features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # Last column (Target)

# Check the target class distribution
print(data['TARGET'].value_counts())

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Visualize the class distribution
sns.countplot(x='TARGET', data=data)
plt.show()

# Check shapes of training and testing sets
print(X_train.shape)
print(X_train.head())
print(y_test.shape)
print(y_test.head())

# Import KNN and train the model
from sklearn.neighbors import KNeighborsClassifier

# Initialize the KNN model
k = 5  # Number of neighbors (You can tune this)
model = KNeighborsClassifier(n_neighbors=k)

# Train the model on the training data
model.fit(X_train, y_train)

# Save the trained model to a file
filename = 'knn_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using accuracy score and confusion matrix
from sklearn import metrics
acc = metrics.accuracy_score(y_pred, y_test)
print("Accuracy is:", acc)
print("Confusion Matrix is:\n", metrics.confusion_matrix(y_pred, y_test))
