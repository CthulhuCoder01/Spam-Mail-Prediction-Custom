# -*- coding: utf-8 -*-
"""Spam-Mail-Prediction-Custom.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FL3XwrmQVQWbJt6k0yH5K8aoxJQ_v1Vn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from scipy.sparse import csr_matrix

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_pred, y):
        epsilon = 1e-10  # small value to avoid log(0)
        loss = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        return loss

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Convert X to dense matrix if it is in csr_matrix format
        if isinstance(X, csr_matrix):
            X = X.toarray()

        # Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent optimization
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Calculate loss and append to loss history
            loss = self.compute_loss(y_pred, y)
            self.loss_history.append(loss)

            # Calculate gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Convert X to dense matrix if it is in csr_matrix format
        if isinstance(X, csr_matrix):
            X = X.toarray()

        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_class = np.where(y_pred > 0.5, 1, 0)
        return y_pred_class

"""Data Collection & Preprocession"""

# Loading The data from cssv file to a pandas data frame
raw_mail_data = pd.read_csv('/content/mail_data.csv')

print(raw_mail_data)

#Now replace all nul value to string using where operator
# when data row is not null do nothing if null change to ''
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

mail_data.head()

#cheack shape of data
mail_data.shape

"""Label encoding change label to numeric outcomes"""

#cheack every row usin loc funciton
mail_data.loc[mail_data['Category']=='spam', 'Category'] = 0
mail_data.loc[mail_data['Category']=='ham', 'Category'] = 1

mail_data.head()

"""Sperating Text and label into seperate array"""

X = mail_data['Message']

Y = mail_data['Category']

print(X)

"""Splitting into training and testing data"""

#Ofcourse use testing and train  split funciton
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

print(X.shape)
print(X_train.shape)
print(X_test.shape)

"""Now comes feature Extraction used to extract feature as numerical data"""

feature_extraction = TfidfVectorizer( min_df=1, stop_words='english', lowercase=True)

# Tranform Using Feature extracion
X_train_feature = feature_extraction.fit_transform(X_train)

X_test_features = feature_extraction.transform(X_test)


# Convert Y_Train and  Y_Test value to string

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(X_train_feature)

"""Training an model"""

model = LogisticRegression()

model.fit(X_train_feature,Y_train)

"""Evaluating Model"""

#Prediction on training data

prediction_on_training_data = model.predict(X_train_feature)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('Accuracy On Training Data :',accuracy_on_training_data)

# Prediction on test data


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print('Accuracy On test Data :',accuracy_on_test_data)

"""Final is to Building a predictive system"""

input_mail = [
    "Nah I don't think he goes to usf, he lives around here though",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
]

# Convert mail text into featurevalue

input_main_features =  feature_extraction.transform(input_mail)

# Making Presictions

prediction = model.predict(input_main_features)

print(prediction)

if prediction[0] == 1 :
  print("Legit mail")
else:
  print("Spam Mail")

if prediction[1] == 1 :
  print("Legit mail Two")
else:
  print("Spam Mail Two")