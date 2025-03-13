import numpy as np
from typing import Optional, List
from ..graph import Graph


class PushRelabelSolver:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.size = graph.size
        self.capacity = graph.capacity
        self.flow = graph.reset_flow()
        self.excess = [0] * self.size
        # « height » représente la hauteur (ou étiquette)
        self.height = [0] * self.size
        self.seen = [0] * self.size

    def push(self, u: int, v: int):
        delta = min(self.excess[u], self.capacity[u, v] - self.flow[u, v])
        self.flow[u, v] += delta
        self.flow[v, u] -= delta
        self.excess[u] -= delta
        self.excess[v] += delta

    def relabel(self, u: int):
        min_height = float('inf')
        for v in range(self.size):
            if self.capacity[u, v] - self.flow[u, v] > 0:
                min_height = min(min_height, self.height[v])
        if min_height < float('inf'):
            self.height[u] = min_height + 1

    def discharge(self, u: int):
        while self.excess[u] > 0:
            if self.seen[u] < self.size:
                v = self.seen[u]
                if self.capacity[u, v] - self.flow[u, v] > 0 and self.height[u] > self.height[v]:
                    self.push(u, v)
                else:
                    self.seen[u] += 1
            else:
                self.relabel(u)
                self.seen[u] = 0

    def solve(self, source: Optional[int] = None, sink: Optional[int] = None) -> int:
        source = source if source is not None else self.graph.source
        sink = sink if sink is not None else self.graph.sink

        self.height[source] = self.size
        self.excess[source] = float('inf')
        for v in range(self.size):
            if v != source:
                self.push(source, v)

        # On exclut la source et le puits de la liste des sommets internes
        vertices = [i for i in range(self.size) if i != source and i != sink]
        p = 0
        while p < len(vertices):
            u = vertices[p]
            old_height = self.height[u]
            self.discharge(u)
            if self.height[u] > old_height:
                # Heuristique "move-to-front"
                vertices.insert(0, vertices.pop(p))
                p = 0
            else:
                p += 1
        return sum(self.flow[source])
