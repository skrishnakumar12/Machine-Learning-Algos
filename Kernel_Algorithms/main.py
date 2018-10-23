import numpy as np
import createsepdata
import kerperceptron
import kerpred
import kerdualsvm


def main():
    X, y = createsepdata.run(10, 2)
    alpha = kerperceptron.run(10, X, y)
    print alpha
    label = kerpred.run(alpha, X, y, np.array([[0], [0]]))
    print label
    newalpha = kerdualsvm.run(X, y)
    print newalpha


if __name__ == "__main__":
    main()
