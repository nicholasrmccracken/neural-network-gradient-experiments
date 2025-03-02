import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split

EPS = 1e-12 # Used to avoid divide by zero errors

# Load spambase data in Q1
url =" http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data,delimiter=",")
x = dataset[:,0:-1]
m = np.median(x, axis = 0)
x = (x>m)*2+(x<=m)*1; # making the feature vectors binary
y = dataset[:,-1]
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size = 0.3, random_state =17)

# Naive Bayes
n = len(y_train)
mu1 = np.mean(x_train[y_train == 1, :], axis=0)
mu2 = np.mean(x_train[y_train == 0, :], axis=0)
var1 = np.var(x_train[y_train == 1, :], axis=0)
var2 = np.var(x_train[y_train == 0, :], axis=0)

# Priors
q1 = np.mean(y_train == 1)
q2 = np.mean(y_train == 0)

def gaussian_nb_likelihood(x_row, mu_vector, var_vector):
    prob = 1.0
    for j in range(x_row.shape[0]):
        denom = np.sqrt(2.0 * np.pi * (var_vector[j] + EPS))
        exponent = np.exp(-0.5 * ((x_row[j] - mu_vector[j])**2) / (var_vector[j] + EPS))
        prob *= (exponent / denom)
    return prob

# Predict labels for test data
predicted_labels = []
for i in range(len(x_test)):
    # Calculate likelihoods (product of univariate Gaussians * priors)
    prob1 = q1 * gaussian_nb_likelihood(x_test[i, :], mu1, var1)
    prob2 = q2 * gaussian_nb_likelihood(x_test[i, :], mu2, var2)
    
    # Choose whichever posterior is larger
    if prob1 >= prob2:
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)

# Calculate test error
predicted_labels = np.array(predicted_labels)
test_error = np.mean(predicted_labels != y_test)
print(f"Test Error: {test_error}")
