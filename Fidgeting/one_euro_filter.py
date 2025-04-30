import math

class LowPassFilter:
    def __init__(self, alpha, initial_value=0.0):
        self.alpha = alpha
        self.s = initial_value
        self.initialized = False

    def filter(self, x):
        if self.initialized:
            self.s = self.alpha * x + (1.0 - self.alpha) * self.s
        else:
            self.s = x
            self.initialized = True
        return self.s

    def set_alpha(self, alpha):
        self.alpha = alpha

class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.lpf_x = LowPassFilter(self.alpha(min_cutoff))
        self.lpf_dx = LowPassFilter(self.alpha(d_cutoff))

    def alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.freq
        dx_hat = self.lpf_dx.filter(dx)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff)
        self.lpf_x.set_alpha(a)
        x_hat = self.lpf_x.filter(x)
        self.x_prev = x
        return x_hat
