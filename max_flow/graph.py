import numpy as np
from scipy.sparse import lil_matrix


class Graph:
    """
    Classe de base pour représenter un réseau de flot avec une matrice creuse.
    """

    def __init__(self, size: int, source: int, sink: int):
        if size <= 0:
            raise ValueError("La taille du graphe doit être positive")

        if not (0 <= source < size) or not (0 <= sink < size):
            raise ValueError(
                "Source et Sink doivent être des indices valides du graphe")

        # Utilisation d'une matrice creuse pour stocker les capacités
        self.capacity = lil_matrix((size, size), dtype=np.float64)
        self.size = size
        self.source = source
        self.sink = sink

    def reset_flow(self):
        """Retourne une matrice de flot initialisée à zéro sous forme creuse."""
        return lil_matrix((self.size, self.size), dtype=np.float64)

    def add_edge(self, u, v, cap):
        """Ajoute une arête dirigée (u -> v) avec une capacité donnée."""
        self.capacity[u, v] = cap
