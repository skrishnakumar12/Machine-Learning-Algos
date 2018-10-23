import numpy as np
import createsepratingdata
import ratingprank
import ratingpred
import ratingsvm
import sol_ratingprank


def main():
    X, y = createsepratingdata.run(10,2,3)
    theta, b = ratingprank.run(10,4,X,y)
    newtheta, newb = sol_ratingprank.run(10, 4, X, y)
    label = ratingpred.run(3, theta, b, np.array([[1], [1]]))
    newtheta, newb = ratingsvm.run(4, X, y)


if __name__ == "__main__":
    main()
