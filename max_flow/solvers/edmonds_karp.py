from collections import deque
import numpy as np
from typing import Optional, List
from ..graph import Graph


class EdmondsKarpSolver:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.size = graph.size
        self.capacity = graph.capacity
        self.flow = graph.reset_flow()

    def bfs(self, source: int, sink: int, parent: List[int]) -> bool:
        visited = [False] * self.size
        queue = deque()
        queue.append(source)
        visited[source] = True

        while queue:
            u = queue.popleft()
            for v in range(self.size):
                # Calcul de la capacité résiduelle
                residual = self.capacity[u, v] - self.flow[u, v]
                if not visited[v] and residual > 0:
                    parent[v] = u
                    visited[v] = True
                    queue.append(v)
                    if v == sink:
                        return True
        return False

    def solve(self, source: Optional[int] = None, sink: Optional[int] = None) -> int:
        source = source if source is not None else self.graph.source
        sink = sink if sink is not None else self.graph.sink

        parent = [-1] * self.size
        max_flow = 0

        while self.bfs(source, sink, parent):
            # Recherche du chemin augmentant et calcul du goulot d'étranglement
            path_flow = float('inf')
            s = sink
            while s != source:
                u = parent[s]
                path_flow = min(
                    path_flow, self.capacity[u, s] - self.flow[u, s])
                s = u
            max_flow += path_flow

            # Mise à jour des flots dans le graphe résiduel
            v = sink
            while v != source:
                u = parent[v]
                self.flow[u, v] += path_flow
                self.flow[v, u] -= path_flow
                v = u

        return max_flow
