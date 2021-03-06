import numpy as np
import string
from collections import Counter


np.random.seed(1)

# Problem 1

def count_frequency(documents):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return frequency

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
def prior_prob(y_train):


    return prior


def conditional_prob(X_train, y_train):



    return cond_prob

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def predict_label(X_test, prior_prob, cond_prob):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return predict, test_prob

def compute_test_prob(word_count, prior_cat, cond_cat):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return prob

def compute_metrics(y_pred, y_true):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return acc, cm, f1


# Problem 2

def featureNormalization(X):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return X_normalized, X_mean, X_std


def applyNormalization(X, X_mean, X_std):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return X_normalized

def computeMSE(X, y, theta):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return error[0]

def computeGradient(X, y, theta):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  return gradient

def gradientDescent(X, y, theta, alpha, num_iters):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return theta, Loss_record

def closeForm(X, y):

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return theta
