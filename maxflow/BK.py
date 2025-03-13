import numpy as np
from collections import deque
from first import Graph


class GraphBK(Graph):
    """
    Implémente l'algorithme de Boykov-Kolmogorov (BK) pour le calcul du flot maximum.
    Hérite de la classe Graph.
    """

    def __init__(self, data: np.ndarray):
        super().__init__(data)
        self.parent = [-1] * self.size  # Stocke le chemin de recherche
        self.tree = [0] * self.size  # 0 = non assigné, 1 = source, 2 = puits

    def _augment(self, bottleneck: int, node: int):
        """Augmente le flot le long du chemin trouvé."""
        while node != self.source:
            prev = self.parent[node]
            self.flow[prev, node] += bottleneck
            self.flow[node, prev] -= bottleneck
            node = prev

    def _grow(self) -> bool:
        """
        Étend les arbres de recherche pour trouver un chemin augmentant.
        Retourne True si un tel chemin est trouvé.
        """
        queue = deque()

        # Initialiser la file avec les sommets sources et puits
        for v in range(self.size):
            if self.tree[v] == 1:  # Appartient à l'arbre source
                queue.append(v)

        while queue:
            node = queue.popleft()

            for neighbor in range(self.size):
                residual = self.capacity[node,
                                         neighbor] - self.flow[node, neighbor]

                if residual > 0:  # Il existe une capacité résiduelle
                    if self.tree[neighbor] == 0:  # Nouveau nœud découvert
                        self.tree[neighbor] = self.tree[node]  # Même arbre
                        self.parent[neighbor] = node
                        queue.append(neighbor)

                    elif self.tree[neighbor] != self.tree[node]:  # Rencontre un autre arbre
                        bottleneck = min(
                            residual,
                            min(self.capacity[self.parent[node], node] - self.flow[self.parent[node], node]
                                for node in queue if node != self.source)
                        )
                        self._augment(bottleneck, node)
                        return True

        return False

    def BK(self) -> int:
        """
        Calcule le flot maximal à l'aide de l'algorithme de Boykov-Kolmogorov.
        """
        self.flow = np.zeros((self.size, self.size),
                             dtype=np.int64)  # Réinitialise le flot

        # Initialisation des arbres de recherche
        self.tree[self.source] = 1  # Arbre source
        self.tree[self.sink] = 2    # Arbre puits

        max_flow = 0

        while self._grow():
            max_flow += 1  # Chaque augmentation augmente le flot total

        return max_flow


if __name__ == "__main__":
    data = np.array([
        [0, 16, 13, 0, 0, 0],
        [0, 0, 10, 12, 0, 0],
        [0, 4, 0, 0, 14, 0],
        [0, 0, 9, 0, 0, 20],
        [0, 0, 0, 7, 0, 4],
        [0, 0, 0, 0, 0, 0]
    ])
    graph = GraphBK(data)
    print(graph.BK())  # 23
