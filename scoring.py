import numpy as np

class QuantileScorer:
    def __init__(self, q: float = 0.9):
        self.q = q

    def score(self, x, y):
        # x: vstup, y: rekonštrukcia (predpokladáme numpy alebo torch tenzory)
        diff = np.abs(x - y)
        return float(np.quantile(diff, q=self.q))