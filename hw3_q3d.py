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

# Training loop
for j in range(1, num_iterations + 1):
    learning_rate = eta / (1 + j * eta)
    indices = np.random.permutation(num_data)

    for i in indices:
        x_i = X_train[i]
        y_i = y_train[i]

        # Compute gradients
        margin_condition = y_i * (np.dot(w, x_i) + b) < 1
        w_grad = (1 / num_data) * w - C * margin_condition * y_i * x_i
        b_grad = -C * margin_condition * y_i
        
        w -= learning_rate * w_grad
        b -= learning_rate * b_grad
        
    # Compute training accuracy
    y_prediction = np.sign(np.dot(X_train, w) + b)
    accuracy = np.mean(y_prediction == y_train)
    train_accuracies.append(accuracy)

# Plot
plt.plot(range(1, num_iterations + 1), train_accuracies)
plt.xlabel("Iteration")
plt.ylabel("Training Accuracy")
plt.title("SVM Batch Gradient Descent: Iteration vs Training Accuracy")
plt.show()
