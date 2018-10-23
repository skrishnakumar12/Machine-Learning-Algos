import numpy as np
# Input: number of labels k
#        vector theta of d rows, 1 column
#        vector b of k-1 rows, 1 column
#        vector x of d rows, 1 column
# Output: label (1 or 2 ... or k)


def run(k,theta,b,x):
    # Your code goes here
    val = np.dot(theta, x)
    if(val <= b[0]):
        label = 1
    for i in range(k - 2):
        if(b[i] < val and val <= b[i+1]):
            label = i + 2
    if(b[k-2] < val):
        label = k
    return label
