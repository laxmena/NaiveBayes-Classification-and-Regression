import numpy as np
import string
from collections import Counter
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

np.random.seed(1)

# Problem 1

def count_frequency(documents):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    lower_case_doc = []
    for doc in documents:
        lower_case_doc.append(doc.lower())
    
    no_punc_doc = []
    for doc in lower_case_doc:
        no_punc_doc.append(doc.translate(str.maketrans('', '', string.punctuation)))
        
    words_doc = []
    for doc in no_punc_doc:
        words_doc += doc.split(' ')
    
    all_words = Counter(words_doc).items()
    frequency = dict(all_words)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return frequency

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
def prior_prob(y_train):
    prior = {}
    
    frequency = Counter(y_train)
    n = len(y_train)
    
    for label in frequency.keys():
        prior[label] = frequency[label] / n
    return prior


def conditional_prob(X_train, y_train):

    alpha, N_alpha = 1, 20000
    
    spam_messages, ham_messages = [], []

    for message, label in zip(X_train, y_train):        
        if(label == 1):
            spam_messages.append(message)
        else:
             ham_messages.append(message)
    
    spam_frequency = count_frequency(spam_messages)
    ham_frequency = count_frequency(ham_messages)
    
    spam_words_count = sum(spam_frequency.values())
    ham_words_count = sum(ham_frequency.values())

    spam_cond_prob, ham_cond_prob = {}, {}
    for word in spam_frequency.keys():
        spam_cond_prob[word] = (spam_frequency[word] + alpha) / (spam_words_count + N_alpha)
    for word in ham_frequency.keys():
        ham_cond_prob[word] = (ham_frequency[word] + alpha) / (ham_words_count + N_alpha)
        
    cond_prob = {}
    cond_prob[0] = ham_cond_prob
    cond_prob[1] = spam_cond_prob

    return cond_prob

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def predict_label(X_test, prior_prob, cond_prob):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    predict, test_prob = [], []
    pred_labels = prior_prob.keys()

    for doc in X_test:
        word_count = count_frequency([doc])
        posterior = [0 for _ in pred_labels]
        
        for i, label in enumerate(pred_labels):            
            posterior[i] += compute_test_prob(word_count, prior_prob[label], cond_prob[label])
            
        max_val = max(posterior)
        predict.append(posterior.index(max_val))
        test_prob.append(list(softmax([val-max_val for val in posterior])))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return predict, test_prob

def compute_test_prob(word_count, prior_cat, cond_cat):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    alpha, N_alpha = 1, 20000
    prob = np.log(prior_cat)
    
    for key, value in word_count.items():
        prob += value * np.log(cond_cat.get(key, (alpha/N_alpha)))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return prob

def compute_metrics(y_pred, y_true):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return acc, cm, f1


# Problem 2

def featureNormalization(X):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X-X_mean)/X_std
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return X_normalized, X_mean, X_std


def applyNormalization(X, X_mean, X_std):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X_normalized = (X-X_mean)/X_std
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return X_normalized

def computeMSE(X, y, theta):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    y_pred = np.sum(theta.T*X, axis=1)
    error = np.sum(np.square(y_pred-y), keepdims=True) / (2*len(y))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return error[0]

def computeGradient(X, y, theta):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    y_pred = np.sum(theta.T*X, axis=1)
    gradient = np.sum((y_pred - y)*X.T, axis=1, keepdims=True) / len(y)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return gradient

def gradientDescent(X, y, theta, alpha, num_iters):

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    Loss_record = np.zeros(num_iters)
    for i in range(num_iters):
        theta -= alpha * computeGradient(X, y, theta)
        Loss_record[i] = computeMSE(X, y, theta)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return theta, Loss_record

def closeForm(X, y):

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    y = y.reshape(-1, 1)
    theta = np.dot(np.linalg.pinv(X.T@X), X.T@y)
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return theta
