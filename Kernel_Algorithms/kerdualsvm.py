import numpy as np
import K
import cvxopt as co
# Input: numpy matrix X of features, with n rows (samples), d columns (features)
#            X[i,j] is the j-th feature of the i-th sample
#        numpy vector y of labels, with n rows (samples), 1 column
#            y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector alpha of n rows, 1 column


def run(X,y):
    # Your code goes here
    rows = X.shape[0] #n
    cols = X.shape[1] #d
    f = np.full(rows, -1)
    b = np.zeros(rows)
    A = np.eye(rows)
    A = np.negative(A)
    H = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            H[i,j] = y[i]*y[j]*K.run(X[i], X[j])
    alpha = np.array(co.solvers.qp(co.matrix(H, tc='d'), co.matrix(f, tc='d'),
                                   co.matrix(A, tc='d'), co.matrix(b, tc='d'))['x'])
    return alpha
