import numpy as np
import K


# Input: number of iterations L
#        numpy matrix X of features, with n rows (samples), d columns (features)
#           X[i,j] is the j-th feature of the i-th sample
#        numpy vector y of labels, with n rows (samples), 1 column
#           y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector alpha of n rows, 1 column


def run(L,X,y):
    # Your code goes here
    alpha = np.zeros((X.shape[0]))
    n = len(y)
    for iterator in range(L):
        for t in range(n):
            s = 0
            for i in range(n):
                s = s + (alpha[i] * y[i] * (K.run(X[i], X[t])))
            if (s*y[t] <= 0):
                alpha[t] = alpha[t] + 1
    return alpha.reshape(-1, 1)
