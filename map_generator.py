import numpy as np
from opensimplex import OpenSimplex
from itertools import count

class MapGenerator:
    def __init__(self, H, W, scale, seed_fn=None):
        self.H = H
        self.W = W
        self.scale = scale
        if seed_fn is not None:
            self.seed_fn = seed_fn
        else:
            self.counter = count()
            self.seed_fn = lambda: next(self.counter)
        
    def __call__(self):
        return self._map_2d()

    def _map_2d(self, threshold=0.75):
        noise = OpenSimplex(seed=self.seed_fn())
        H, W, scale = self.H, self.W, self.scale
        out = np.empty((H, W))
        for x in np.arange(W):
            for y in np.arange(H):
                value = (noise.noise2d(x / scale, y / scale) + 1)/2
                out[x, y] = value
        return out > threshold