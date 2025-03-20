import numpy as np


class Graph:
    """
    Classe de base pour représenter un réseau de flot.
    Stocke la matrice des capacités et quelques informations de base.
    """

    def __init__(self, capacity: np.ndarray):
        if capacity is None:
            raise ValueError("La matrice de capacité doit être fournie")

        n, m = capacity.shape
        if n != m:
            raise ValueError("La matrice de capacité doit être carrée")

        # On copie la matrice pour éviter les modifications inattendues
        self.capacity = capacity.copy()
        self.size = n
        self.source = 0
        self.sink = n - 1

    def reset_flow(self) -> np.ndarray:
        """
        Retourne une matrice de flot initialisée à zéro.
        """
        return np.zeros((self.size, self.size), dtype=np.int64)
