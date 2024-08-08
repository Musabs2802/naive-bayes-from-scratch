# Naive Bayes
# The Naive Bayes classifier is a probabilistic classifier based on applying Bayes theorem with strong(naive) independence assumptions between features
# y = argmax[log(P(x1|y)) + log(P(x2|y)) + ... + log(P(xn|y)) + log(P(y))]
# 
# P(y) -> Prior probablility: Frequency of each class
# P(xi|y) -> Class conditional probablity: Model with Gaussian
# 
# Steps
# Training:
# Calculate mean, var and priors of each class
# 
# Testing:
# Calculate posterior for each class with
# y = argmax[log(P(x1|y)) + log(P(x2|y)) + ... + log(P(xn|y)) + log(P(y))]
# and Gaussian formula
# Choose class with highest posterior probablilty

import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Calculate mean, var, priors for each class
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i] = X_c.mean(axis=0)
            self.var[i] = X_c.var(axis=0)
            self.priors[i] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        return np.array([self.__predict(x) for x in X])
    
    def __predict(self, x):
        posteriors = []

        for i, _ in enumerate(self.classes):
            prior = np.log(self.priors[i])
            posterior = np.sum(np.log(self.__pxi_y(i, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]
    
    def __pxi_y(self, idx, x):
        mean = self.mean[idx]
        var = self.var[idx]
        return  np.exp(-(x - mean)**2 / (2*var)) / np.sqrt(2 * np.pi * var)
