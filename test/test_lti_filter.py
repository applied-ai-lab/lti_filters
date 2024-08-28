import numpy as np
from matplotlib import pyplot as plt

from lti_filters.discrete_filters import (LTIBaseParams, DiscreteManualLTI)


def test():

    rise_time = 1.0      # Peak time occurs at 0.1 seconds
    damping_ratio = 0.7 # Critical damping example

    params = LTIBaseParams(rise_time, damping_ratio)

    dim = 2
    dt = 0.0025
    filter = DiscreteManualLTI(dim, params, dt)

    no_iters = int((3 * rise_time) / dt)

    # Create a step input 
    u = np.ones([dim, no_iters])

    # Set the first element of u to zero
    u[:, 0] *= 0.0

    y = np.zeros([dim, no_iters])

    for k in range(no_iters):
        y[:, k] = filter.advance(u[:, k])

    
    plt.figure()
    plt.plot(u[0, :])
    plt.plot(y[0, :])
    plt.show()

    return 0


if __name__ == "__main__":
    test()
