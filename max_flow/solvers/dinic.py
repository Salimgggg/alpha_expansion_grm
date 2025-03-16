import numpy as np
from collections import defaultdict


class DinicSolver:
    def __init__(self, graph):
        self.graph = graph
        self.size = graph.capacity.shape[0]
        self.source = graph.source
        self.sink = graph.sink
        self.level = np.full(self.size, -1, dtype=int)
        self.iter = np.zeros(self.size, dtype=int)
        self.flow = defaultdict(dict)

    def bfs_level(self):
        queue = [self.source]
        self.level.fill(-1)
        self.level[self.source] = 0

        for u in queue:
            neighbors = self.graph.capacity[u].nonzero()[1]
            for v in neighbors:
                if self.level[v] == -1 and self.graph.capacity[u, v] > self.flow.get(u, {}).get(v, 0):
                    self.level[v] = self.level[u] + 1
                    queue.append(v)

        return self.level[self.sink] != -1

    def send_flow(self, u, flow):
        if u == self.sink:
            return flow

        neighbors = self.graph.capacity[u].nonzero()[1]
        while self.iter[u] < len(neighbors):
            v = neighbors[self.iter[u]]
            residual = self.graph.capacity[u, v] - \
                self.flow.get(u, {}).get(v, 0)

            if self.level[v] == self.level[u] + 1 and residual > 0:
                min_flow = min(flow, residual)
                pushed = self.send_flow(v, min_flow)

                if pushed > 0:
                    self.flow[u][v] = self.flow.get(u, {}).get(v, 0) + pushed
                    self.flow[v][u] = self.flow.get(v, {}).get(u, 0) - pushed
                    return pushed

            self.iter[u] += 1
        return 0

    def solve(self):
        max_flow = 0
        while self.bfs_level():
            self.iter.fill(0)
            while flow := self.send_flow(self.source, float('inf')):
                max_flow += flow
        return max_flow
