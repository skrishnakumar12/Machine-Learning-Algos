import numpy as np
# Input: number of iterations L
#        number of labels k
#        matrix X of features, with n rows (samples), d columns (features)
#            X[i,j] is the j-th feature of the i-th sample
#        vector y of labels, with n rows (samples), 1 column
#            y[i] is the label (1 or 2 ... or k) of the i-th sample
# Output: vector theta of d rows, 1 column
#         vector b of k-1 rows, 1 column


def run(L,k,X,y):
    # Your code goes here
    n = len(X)
    d = len(X[0])
    theta = np.zeros(d)
    b = np.zeros(k-1)
    s = np.zeros((n, k-1))

    for l in range(0, k-1):
        b[l] = l

    for t in range(0,n):
        for l in range(0, k-1):
            if(y[t] <= l + 1):
                s[t][l] = -1
            else:
                s[t][l] = 1

    for iterator in range(0, L):
        for t in range(0, n):
            E = np.zeros(k-1)
            for l in range(0, k-1):
                if((s[t][l]*(np.dot(theta, X[t]) - b[l])) <= 0):
                    E[l] = l
            if(sum(E) > 0):
                newsum = 0
                for m in range(0, k-1):
                    if(E[m] > 0):
                        newsum += s[t][int(E[m])]
                for a in range(0, d):
                    theta[a] += newsum*X[t][a]
                for x in range(0, k-1):
                    if(E[x] > 0):
                        b[int(E[x])] -= s[t][int(E[x])]
    return theta, b
