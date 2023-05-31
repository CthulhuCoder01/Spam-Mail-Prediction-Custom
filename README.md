# Logistic Regression for Spam Classification with my own Logistic Regression Model Class from scratch

## Project Description

This project focuses on using logistic regression, a popular machine learning algorithm, for classifying emails as either spam or legitimate. Logistic regression is a binary classification algorithm that estimates the probability of an email being spam based on its features.

## Features

- **Data Collection:** Gather a labeled dataset of emails, where each email is tagged as spam or legitimate. This dataset will serve as the training data for our logistic regression model.

- **Data Preprocessing:** Perform necessary preprocessing steps on the email data, including removing stop words, converting text to lowercase, and handling special characters or formatting issues. This step ensures that the data is in a suitable format for training the model.

- **Feature Extraction:** Extract relevant features from the email data to represent them numerically. This can include techniques like bag-of-words, TF-IDF (Term Frequency-Inverse Document Frequency), or word embeddings to capture the semantic meaning of words.

- **Model Training:** Created Custom Logistic Algorithm with 0.86 Accuracy. Split the preprocessed dataset into training and testing sets. Use the training set to train the logistic regression model, optimizing the model's parameters to minimize the error between predicted and actual labels.

- **Model Evaluation:** Evaluate the trained model's performance using the testing set. Common evaluation metrics for binary classification include accuracy, precision, recall, and F1 score. These metrics provide insights into the model's ability to correctly classify spam and legitimate emails.

- **Prediction:** Once the logistic regression model is trained and evaluated, apply it to new, unseen emails to predict whether they are spam or legitimate. This prediction step helps in automating the classification process for future email filtering or sorting tasks.


## Logistic Regression Explanation
```code
import numpy as np
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

```

## Explanation:

  1. The code defines a LogisticRegression class that implements logistic regression for binary classification.
  2. The class has an initialization method __init__ that sets the learning rate, number of iterations, weights, bias, and loss history.
  3. The sigmoid method calculates the sigmoid function, which maps any real number to the range [0, 1].
  ![equation](https://latex.codecogs.com/png.latex?\frac{1}{1+e^{m}})
  5. The compute_loss method calculates the cross-entropy loss between the predicted values (y_pred) and the actual values (y).
  6. The fit method trains the logistic regression model using gradient descent optimization. It initializes the weights and bias, performs the optimization loop for the specified number of iterations, calculates the loss at each iteration, computes the gradients, and updates the weights and bias accordingly.
  7. The predict method takes an input matrix X, applies the learned weights and bias to calculate the linear model, applies the sigmoid function to obtain predicted probabilities, and converts the probabilities to binary predictions based on a threshold of 0.5.



## Conclusion

Logistic regression is a powerful algorithm for email classification, enabling the differentiation between spam and legitimate emails based on their features. By implementing logistic regression and leveraging its probabilistic nature, we can effectively classify emails and enhance email management, spam filtering, and user experience.

## This Project also contains the trained model

