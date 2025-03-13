from collections import deque
import numpy as np
from typing import Optional, List
from ..graph import Graph


class DinicSolver:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.size = graph.size
        self.capacity = graph.capacity
        self.flow = graph.reset_flow()
        self.level = [-1] * self.size

    def bfs_level(self, source: int, sink: int) -> bool:
        self.level = [-1] * self.size
        self.level[source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in range(self.size):
                if self.level[v] < 0 and self.capacity[u, v] - self.flow[u, v] > 0:
                    self.level[v] = self.level[u] + 1
                    queue.append(v)
        return self.level[sink] >= 0

    def send_flow(self, u: int, sink: int, flow: int, start: List[int]) -> int:
        if u == sink:
            return flow
        for v in range(start[u], self.size):
            start[u] = v
            if self.level[v] == self.level[u] + 1 and self.capacity[u, v] - self.flow[u, v] > 0:
                curr_flow = min(flow, self.capacity[u, v] - self.flow[u, v])
                temp_flow = self.send_flow(v, sink, curr_flow, start)
                if temp_flow > 0:
                    self.flow[u, v] += temp_flow
                    self.flow[v, u] -= temp_flow
                    return temp_flow
        return 0

    def solve(self, source: Optional[int] = None, sink: Optional[int] = None) -> int:
        source = source if source is not None else self.graph.source
        sink = sink if sink is not None else self.graph.sink
        max_flow = 0

        while self.bfs_level(source, sink):
            start = [0] * self.size
            while True:
                flow = self.send_flow(source, sink, float('inf'), start)
                if flow <= 0:
                    break
                max_flow += flow
        return max_flow
