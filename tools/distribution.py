import numpy as np


class Distribution(object):
    pass


class RandomDistribution(Distribution):
    def __init__(self, max_range):
        super().__init__()
        self.max_range = max_range

    def sample(self, n: list):
        # time = time % self.max_time
        values = [None] * len(n)
        for i, e in enumerate(n):
            # values[i] = np.random.randint(0, self.max_range, e)
            # assert e >= 0
            values[i] = np.random.choice(self.max_range, e)
        return values


class LimitLayers(Distribution):
    def __init__(self, w_limit: int, h_limit: int, layer_limit: int):
        self._width = w_limit
        self._height = h_limit
        self._max_layer = layer_limit

    def sample(self, n: list, focus_grid: list):
        values = [None] * len(n)

        for i, e in enumerate(n):
            r, c = focus_grid[i] // self._width, focus_grid[i] % self._width
            neighbor_list = self._get_neighbor(r, c)
            values[i] = np.random.choice(neighbor_list, e)
        return values

    def _get_neighbor(self, r, c):
        l_r = max(r - self._max_layer, 0)
        r_r = min(r + self._max_layer, self._height - 1)
        l_c = max(0, c - self._max_layer)
        r_c = min(c + self._max_layer, self._width - 1)

        neighbors = []
        for _r in range(l_r, r_r):
            _index_r = _r * self._width
            for _c in range(l_c, r_c):
                neighbors.append(_index_r + _c)
        return neighbors

class GaussianDistribution(Distribution):
    def __init__(self, width: int, height: int, n_peak: int, max_range=10, max_time=200, sigma=None):
        super().__init__()
        self.max_range = max_range  # control the maximum of value
        self.n_model = n_peak  # the amount of single model
        self._max_step = height * width  # limit the maximum of steps
        self.width, self.height = width, height

        self._pos = np.zeros((width, height, 2))
        # self._direct = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        # self._direct = [[1, 1], [-1, -1]]
        self._direct = [[1, 1], [-1, 0], [1, -1], [-1, 0]]

        x, y = np.meshgrid(np.arange(width), np.arange(height))
        self._pos[:, :, 0] = x / width
        self._pos[:, :, 1] = y / height

        # print('shape of pos:', self._pos.shape)

        self._x, self._y, self._flag = 0, 0, -1

        self.sigma = sigma

    def _multivariate_gaussian(self, pos, mu, sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.
        """

        n = mu.shape[0]
        sigma_det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        N = np.sqrt((2*np.pi)**n * sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos - mu, sigma_inv, pos - mu)

        return np.exp(-1.0 * fac / 2.) / N

    def mu_offset(self, shape, seed=0):
        np.random.seed(seed)
        return np.random.random_integers(0, self.width, shape)

    def sigma_offset(self, shape, seed=0):
        np.random.seed(seed)
        return np.random.randn(*shape)

    def sample(self, time, seed=0):
        time = time % self._max_step
        # mu, sigma = self.mu_dist[time], self.sigma_dist[time]
        # mu = self.mu_offset((self.n_model, 2), time) / self.width
        mu = self._pos[self._x, self._y, :].reshape((1, -1))
        # print(self._x, self._y, mu)
        # sigma = self.sigma_offset((self._max_step, 2), time)
        sigma = np.array([[0., 0.25], [0., 0.25]]) if self.sigma is None else self.sigma

        # shape_mu, shape_sigma = np.array(mu).shape, np.array(sigma).shape
        prob = np.zeros(self._pos.shape[:2])

        for i in range(self.n_model):
            # sigma = sigma + self.sigma_offset(shape_sigma, i)
            temp_prob = self._multivariate_gaussian(self._pos, mu[i], sigma)
            prob += temp_prob

        prob /= np.sum(prob)
        value = self.max_range * prob
        value = value.reshape((-1,))
        # print(value.astype(np.int32))

        if self._overflow(self._direct[self._flag], [self.width, self.height]):
            self._flag = (self._flag + 1) % len(self._direct)

        _direct = self._direct[self._flag]
        self._x += _direct[0]
        self._y += _direct[1]
    
        return value.astype(np.int32)
    
    def _overflow(self, direct, max_value):
        a = self._x + direct[0]
        b = self._y + direct[1]

        return (a < 0 or a >= max_value[0]) or (b < 0 or b >= max_value[1])


class Normal(Distribution):
    def __init__(self, file, width: int, height: int, max_time=100, scale=1.):
        import pickle
        self.dist = []
        self._time = 0
        self._max_time = 100
        self.width = width
        self.height = height

        with open(file, 'rb') as f:
            dist = pickle.load(f)

        # for value in dist.values():
        #     self.dist.append(value)
        self.dist = dist
        self.scale = scale
        self._max_time = len(self.dist[0])

    def sample(self, time, seed=0):
        time = time % self._max_time
        value = np.zeros(self.width * self.height, dtype=np.int32)

        for i in range(self.width * self.height):
            mu, sigma = self.dist[i][time]
            mu *= self.scale
            value[i] = int(max(np.random.normal(mu, sigma), 0.) + 1)

        return value
