import numpy as np
import matplotlib.pyplot as plt

# Load training data
X_train = np.loadtxt('digits_training_data.csv', delimiter=',')
y_train = np.loadtxt('digits_training_labels.csv')
y_train = np.where(y_train == 4, -1, 1)

# Initialize parameters
num_data, num_features = X_train.shape
w = np.zeros(num_features)
b = 0

# Hyperparameters
C = 3
eta = 0.001
num_iterations = 40
train_accuracies = []

# Compute gradients for svm batch gradient descent
def compute_gradients(X, y, w, b, C):
    margin_condition = y * (X @ w + b) < 1
    w_grad = w / num_data - C * np.sum(
        (y[margin_condition, None] * X[margin_condition]), 
        axis=0)
    b_grad = -C * np.sum(y[margin_condition])
    return w_grad, b_grad

# Training loop
for j in range(1, num_iterations + 1):
    learning_rate = eta / (1 + j * eta)
    w_grad, b_grad = compute_gradients(X_train, y_train, w, b, C)
    
    w -= learning_rate * w_grad
    b -= learning_rate * b_grad
    
    # Compute training accuracy
    y_prediction = np.sign(X_train @ w + b)
    accuracy = np.mean(y_prediction == y_train)
    train_accuracies.append(accuracy)

# Plot
plt.plot(range(1, num_iterations + 1), train_accuracies)
plt.xlabel("Iteration")
plt.ylabel("Training Accuracy")
plt.title("SVM Batch Gradient Descent: Iteration vs Training Accuracy")
plt.show()
