import numpy as np
from scipy import signal

from collections import namedtuple

LTIOutput = namedtuple('LTIOutput', ['t_out', 'y', 'x'])


class LTIBaseParams:
    def __init__(self, peak_time, damping_ratio):
        self.desired_peak_time = peak_time
        self.damping_ratio = damping_ratio


class LTIfilter:
    def __init__(self, params: LTIBaseParams):
        self.params = params

        self.lti = self.create_lti()

    def create_lti(self):
        # Use HLT page 168 figure to work out w0
        w0 = self.calc_w0(self.params.desired_peak_time, self.params.damping_ratio)

        num = [w0 ** 2]
        den = [1, 2 * self.params.damping_ratio * w0, w0 ** 2]

        return signal.lti(num, den)

    @staticmethod
    def calc_w0(peak_time, damping_ratio):
        if damping_ratio < 1.0:
            return np.pi / (peak_time * np.sqrt(1 - damping_ratio ** 2.0))
        else:
            return 3.0 / peak_time

    def get_output(self, u_arr, t_arr, x_init=None):
        return LTIOutput(* self.lti.output(u_arr, t_arr, x_init))


class DiscreteManualLTI(LTIfilter):
    def __init__(self, dim: int, params: LTIBaseParams, dt=1.0/400.0):
        super().__init__(params)

        self.N = dim

        self.dt = dt

        self.state_space = self.lti.to_discrete(dt, method='impulse').to_ss()

        self.A = np.stack(list(self.state_space.A for _ in range(self.N)))
        self.B = np.stack(list(self.state_space.B for _ in range(self.N)))
        self.C = np.stack(list(self.state_space.C for _ in range(self.N)))
        self.D = np.stack(list(self.state_space.D for _ in range(self.N)))

        self.x_k = np.zeros([self.N, self.state_space.A.shape[1], 1])
        self.x_k1 = np.zeros([self.N, self.state_space.A.shape[1], 1])
        self.u_k = np.zeros([self.N, self.state_space.B.shape[1], 1])
        self.y_k1 = np.zeros([self.N, self.state_space.D.shape[0], self.state_space.D.shape[1]])

    def reset(self, x_init=None, u_init=None):
        if x_init is not None:
            self.x_k = x_init
            self.x_k1 = x_init
        else:
            self.x_k *= 0.0
            self.x_k1 *= 0.0

        if u_init is not None:
            if len(self.u_k.shape) == len(u_init.shape):
                self.u_k = u_init
            else:
                self.u_k[:, 0, 0] = u_init
        else:
            self.u_k *= 0.0

        self.y_k1 *= 0.0
        return

    def initialise(self, iters=1000, x_init=None, u_init=None):
        self.reset(x_init, u_init)
        for _ in range(iters):
            self.advance(self.u_k)
        return

    def advance(self, u_k):
        if len(u_k.shape) == len(self.u_k.shape):
            self.u_k = u_k
        else:
            self.u_k[:, 0, 0] = u_k 

        self.x_k1 = np.matmul(self.A, self.x_k) + np.matmul(self.B, self.u_k)
        self.y_k1 = np.matmul(self.C, self.x_k1) + self.D
        self.x_k = self.x_k1
        return self.y_k1.squeeze()
