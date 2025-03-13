import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any


class Graph:
    """
    A flow network graph implementation for maximum flow algorithms.

    This class implements various maximum flow algorithms:
    - Edmonds-Karp algorithm
    - Dinic's algorithm
    - Push-Relabel algorithm

    Attributes:
        capacity: Adjacency matrix representing edge capacities
        flow: Adjacency matrix representing current flow
        size: Number of vertices in the graph
        vertices: Array of vertex indices
        excess: List of excess flow at each vertex
        distance: List of distance labels for each vertex
        level: List of level/layer for each vertex (used in Dinic's algorithm)
        seen: List tracking seen vertices during algorithm execution
        source: Source vertex index (default is 0)
        sink: Sink vertex index (default is size-1)
    """

    def __init__(self, data: Optional[np.ndarray] = None):
        """
        Initialize a flow network graph.

        Args:
            data: Adjacency matrix representing edge capacities.
                 Must be a square matrix.

        Raises:
            AssertionError: If data is not a square matrix or if source/sink
                           are not properly configured.
        """
        if data is None:
            raise ValueError("Capacity matrix must be provided")

        n, m = np.shape(data)
        assert n == m, "Capacity matrix must be square"

        # Edge properties
        self.capacity = data
        self.flow = np.zeros((n, n), dtype=np.int64)

        self.size = n

        # Vertex properties
        self.vertices = np.arange(self.size)
        self.excess = [0] * self.size
        self.distance = [0] * self.size
        self.level = [-1] * self.size
        self.seen = [0] * self.size

        # Validate source and sink
        assert np.all(
            self.capacity[-1, :] == 0), "Sink (last node) should not have outgoing edges."
        assert np.all(
            self.capacity[:, 0] == 0), "Source (first node) should not have incoming edges."

        self.source = 0
        self.sink = self.size - 1

    def _search(self, traverse: List[int], origin: Optional[int] = None,
                goal: Optional[int] = None, DFS: bool = False) -> bool:
        """
        Find an augmenting path between given origin and goal using DFS or BFS.

        Args:
            traverse: A list that holds the traverse path where traverse[successor] = current
            origin: Starting vertex index (default is source)
            goal: Target vertex index (default is sink)
            DFS: If True, use Depth-First Search; if False, use Breadth-First Search

        Returns:
            True if a path exists from origin to goal, False otherwise
        """
        # Set path origin and goal indexes
        goal = goal or self.sink
        origin = origin or self.source

        # Track visited vertices
        visited = [False] * self.size

        # Queue of vertices to visit, starting with origin
        to_visit = [origin]

        # Mark origin as visited
        visited[origin] = True

        while to_visit:
            # Select next vertex based on search method
            if DFS:
                current = to_visit.pop(0)  # DFS uses a stack (LIFO)
            else:
                current = to_visit.pop(0)  # BFS uses a queue (FIFO)

            # Explore all possible edges from current vertex
            for successor in range(self.size):
                # Check if edge has residual capacity and successor not visited
                residual = self.capacity[current,
                                         successor] - self.flow[current, successor]
                if not visited[successor] and residual > 0:
                    # Mark as visited and add to exploration queue
                    visited[successor] = True
                    to_visit.append(successor)

                    # Record the path
                    traverse[successor] = current

        # Return whether the goal was reached
        return visited[goal]

    def _max_flow_search_FF(self, origin: Optional[int] = None,
                            goal: Optional[int] = None,
                            data: bool = False,
                            DFS: bool = False) -> Union[int, Dict[str, Any]]:
        """
        Find maximum flow using Ford-Fulkerson method with either BFS or DFS.

        This is the core algorithm used by Edmonds-Karp (when BFS=False).

        Args:
            origin: Starting vertex index (default is source)
            goal: Target vertex index (default is sink)
            data: If True, return detailed iteration data
            DFS: If True, use Depth-First Search; if False, use Breadth-First Search

        Returns:
            Either the maximum flow value or a dictionary with flow and iteration data

        Time Complexity: 
            O(V·E²) where V is the number of vertices and E is the number of edges
        """
        goal = goal or self.sink
        origin = origin or self.source

        # For storing detailed iteration data if requested
        presented_data = {}

        # Initialize variables
        traverse = [-1] * self.size
        max_flow = 0
        iter_count = 0

        # Continue until no more augmenting paths exist
        while self._search(traverse, origin, goal, DFS):
            iter_count += 1

            if data:
                presented_data[iter_count] = []

            # Find the bottleneck capacity (minimum residual capacity along the path)
            path_flow = float('inf')
            current = goal

            while current != origin:
                if data:
                    presented_data[iter_count].append(current)

                pred = traverse[current]

                # Calculate residual capacity of the edge
                residual_capacity = self.capacity[pred,
                                                  current] - self.flow[pred, current]
                path_flow = min(path_flow, residual_capacity)

                current = pred

            # Augment flow along the path
            max_flow += path_flow

            current = goal
            while current != origin:
                pred = traverse[current]

                # Update flow values
                self.flow[pred, current] += path_flow
                # Reverse edge for residual graph
                self.flow[current, pred] -= path_flow

                current = pred

        # Return results based on requested format
        if data:
            return {
                'max_flow': max_flow,
                'iteration': [{i: [0] + presented_data[i][::-1]} for i in range(1, iter_count+1)]
            }

        return max_flow

    def EdmondKarp(self, origin: Optional[int] = None,
                   goal: Optional[int] = None,
                   data: bool = False) -> Union[int, Dict[str, Any]]:
        """
        Compute maximum flow using the Edmonds-Karp algorithm.

        This algorithm uses BFS to find the shortest augmenting path from source to sink
        in each iteration. For each path, it finds the bottleneck capacity and augments
        the flow accordingly. This repeats until no more augmenting paths exist.

        Args:
            origin: Starting vertex index (default is source)
            goal: Target vertex index (default is sink)
            data: If True, return detailed iteration data

        Returns:
            Either the maximum flow value or a dictionary with flow and iteration data

        Time Complexity: 
            O(V·E²) where V is the number of vertices and E is the number of edges

        Notes:
            The algorithm is guaranteed to terminate because the length of the
            shortest augmenting path increases monotonically, and there can be
            at most V such increases.
        """
        self.flow = np.zeros((self.size, self.size),
                             dtype=np.int64)  # Reset flow
        return self._max_flow_search_FF(origin=origin, goal=goal, data=data, DFS=False)

    def _BFS_using_levels(self, origin: Optional[int] = None,
                          goal: Optional[int] = None) -> List[int]:
        """
        Perform BFS to assign level/distance values to vertices.

        Used in Dinic's algorithm to build the level graph.

        Args:
            origin: Starting vertex index (default is source)
            goal: Target vertex index (default is sink)

        Returns:
            List of vertex levels, where level[v] is the shortest distance from origin to v
        """
        origin = origin or self.source
        goal = goal or self.sink

        # Initialize all vertices with level -1 (unreachable)
        self.level = [-1] * self.size
        self.level[origin] = 0  # Origin has level 0

        to_visit = [origin]

        while to_visit:
            current = to_visit.pop(0)

            for successor in range(self.size):
                # Check if edge has residual capacity and successor not leveled
                residual = self.capacity[current,
                                         successor] - self.flow[current, successor]
                if residual > 0 and self.level[successor] < 0:
                    self.level[successor] = self.level[current] + 1
                    to_visit.append(successor)

        return self.level

    def _send_flow(self, origin: Optional[int] = None,
                   goal: Optional[int] = None,
                   max_flow: int = float('inf')) -> int:
        """
        Recursively send flow through the level graph from origin to goal.

        Used in Dinic's algorithm to find blocking flows.

        Args:
            origin: Starting vertex index (default is source)
            goal: Target vertex index (default is sink)
            max_flow: Maximum flow allowed (bottleneck so far)

        Returns:
            Amount of flow sent through the path, or 0 if no path exists
        """
        goal = goal or self.sink
        origin = origin or self.source

        # If reached the sink, return the bottleneck flow
        if origin == goal:
            return max_flow

        # Try to send flow through each edge
        for successor in range(self.size):
            # Check residual capacity and level constraint
            residual = self.capacity[origin,
                                     successor] - self.flow[origin, successor]
            if residual > 0 and self.level[origin] + 1 == self.level[successor]:
                # Calculate bottleneck for this path
                current_flow = min(max_flow, residual)

                # Recursively try to send flow from successor to sink
                path_flow = self._send_flow(successor, goal, current_flow)

                # If flow was sent, update the flow values and return
                if path_flow > 0:
                    self.flow[origin, successor] += path_flow
                    self.flow[successor, origin] -= path_flow
                    return path_flow

        return 0

    def _max_flow_search_D(self, origin: Optional[int] = None,
                           goal: Optional[int] = None) -> int:
        """
        Find maximum flow using Dinic's algorithm.

        Args:
            origin: Starting vertex index (default is source)
            goal: Target vertex index (default is sink)

        Returns:
            Maximum flow value
        """
        goal = goal or self.sink
        origin = origin or self.source
        total_flow = 0

        # Continue until no more augmenting paths exist
        while True:
            # Construct the level graph
            self.level = self._BFS_using_levels(origin, goal)

            # If sink is not reachable, we're done
            if self.level[goal] < 0:
                return total_flow

            # Find blocking flows in the level graph
            while True:
                path_flow = self._send_flow(origin, goal)

                # If no more flow can be sent, break
                if path_flow == 0:
                    break

                # Add to total flow
                total_flow += path_flow

    def Dinic(self, origin: Optional[int] = None,
              goal: Optional[int] = None) -> int:
        """
        Compute maximum flow using Dinic's algorithm.

        This algorithm constructs a level graph using BFS, then finds blocking
        flows in this graph using multiple DFS searches. It repeats this process
        until no more augmenting paths exist.

        Args:
            origin: Starting vertex index (default is source)
            goal: Target vertex index (default is sink)

        Returns:
            Maximum flow value

        Time Complexity: 
            O(V²·E) where V is the number of vertices and E is the number of edges

        Notes:
            Dinic's algorithm generally outperforms Edmonds-Karp for dense graphs.
            Each iteration finds longer paths than the previous one, and there
            can be at most V iterations.
        """
        self.flow = np.zeros((self.size, self.size),
                             dtype=np.int64)  # Reset flow
        return self._max_flow_search_D(origin=origin, goal=goal)

    def _push(self, u: int, v: int) -> None:
        """
        Push excess flow from vertex u to vertex v.

        Used in the Push-Relabel algorithm.

        Args:
            u: Source vertex for the push operation
            v: Target vertex for the push operation
        """
        # Calculate amount of flow to push (minimum of excess and residual capacity)
        delta_flow = min(self.excess[u],
                         self.capacity[u, v] - self.flow[u, v])

        # Update flow and excess values
        self.flow[u, v] += delta_flow
        self.flow[v, u] -= delta_flow

        self.excess[u] -= delta_flow
        self.excess[v] += delta_flow

    def _relabel(self, u: int) -> None:
        """
        Relabel vertex u by increasing its distance label.

        Used in the Push-Relabel algorithm when no valid push operation is possible.

        Args:
            u: Vertex to relabel
        """
        # Find the minimum distance of all adjacent vertices with residual capacity
        min_distance = float('inf')

        for v in range(self.size):
            residual = self.capacity[u, v] - self.flow[u, v]
            if residual > 0:
                min_distance = min(min_distance, self.distance[v])

        # Relabel u to be one more than the minimum distance found
        if min_distance != float('inf'):
            self.distance[u] = min_distance + 1

    def _discharge(self, u: int) -> None:
        """
        Discharge excess flow from vertex u using push and relabel operations.

        Used in the Push-Relabel algorithm to handle active vertices.

        Args:
            u: Vertex to discharge
        """
        # Continue until u has no excess flow
        while self.excess[u] > 0:
            # If we haven't tried all neighbors
            if self.seen[u] < self.size:
                v = self.seen[u]

                # Check if push is valid: residual capacity exists and u is higher than v
                residual = self.capacity[u, v] - self.flow[u, v]
                if residual > 0 and self.distance[u] > self.distance[v]:
                    # Perform push operation
                    self._push(u, v)
                else:
                    # Try next neighbor
                    self.seen[u] += 1
            else:
                # No valid push possible, relabel u and reset neighbor counter
                self._relabel(u)
                self.seen[u] = 0

    def PushRelabel(self, origin: Optional[int] = None,
                    goal: Optional[int] = None) -> int:
        """
        Compute maximum flow using the Push-Relabel algorithm.

        This algorithm works by maintaining a valid preflow and gradually
        converting it to a maximum flow. It makes local decisions by repeatedly
        selecting an overflowing vertex and discharging it through push and
        relabel operations.

        Args:
            origin: Starting vertex index (default is source)
            goal: Target vertex index (default is sink)

        Returns:
            Maximum flow value

        Time Complexity: 
            O(V³) where V is the number of vertices

        Notes:
            The Push-Relabel algorithm often outperforms Dinic's algorithm
            for dense graphs. This implementation uses the "relabel-to-front"
            selection rule for better performance.
        """
        # Set default values
        goal = goal or self.sink
        origin = origin or self.source

        # Identify internal nodes (not source or sink)
        inter_nodes = [i for i in range(
            self.size) if i != origin and i != goal]

        # Initialize preflow: set distance labels and saturate edges from source
        self.distance[origin] = self.size
        self.excess[origin] = float('inf')

        # Push flow from source to all neighbors
        for v in range(self.size):
            self._push(origin, v)

        # Process vertices until no active vertices remain
        potential_vertex = 0
        while potential_vertex < len(inter_nodes):
            u = inter_nodes[potential_vertex]

            # Save previous distance for relabel detection
            prev_distance = self.distance[u]

            # Discharge excess flow from u
            self._discharge(u)

            # Check if vertex was relabeled
            if self.distance[u] > prev_distance:
                # Move-to-front heuristic: prioritize recently relabeled vertices
                inter_nodes.insert(0, inter_nodes.pop(potential_vertex))
                potential_vertex = 0
            else:
                # Move to next vertex
                potential_vertex += 1

        # Total flow equals flow leaving the source
        return sum(self.flow[origin])


def test_graph():
    # Test Case 1: Simple Graph
    capacity_matrix = np.array([
        [0, 10, 10, 0, 0, 0],
        [0, 0, 2, 4, 8, 0],
        [0, 0, 0, 0, 9, 0],
        [0, 0, 0, 0, 0, 10],
        [0, 0, 0, 6, 0, 10],
        [0, 0, 0, 0, 0, 0]
    ])

    graph = Graph(capacity_matrix)

    # Expected max flow: 19 (10 through node 2, 9 through node 1)

    assert graph.Dinic() == 19, "Dinic's algorithm failed"
    assert graph.EdmondKarp() == 19, "Edmonds-Karp algorithm failed"

    print("Test Case 1 passed")

    # Test Case 2: Small Graph with One Path
    capacity_matrix = np.array([
        [0, 5, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4],
        [0, 0, 0, 0]
    ])

    graph = Graph(capacity_matrix)

    # Expected max flow: 3 (bottleneck at edge (1,2))
    assert graph.EdmondKarp() == 3, "Edmonds-Karp algorithm failed"
    assert graph.Dinic() == 3, "Dinic's algorithm failed"

    print("Test Case 2 passed")

    # Test Case 3: Disconnected Graph (No Flow)
    capacity_matrix = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    graph = Graph(capacity_matrix)

    # Expected max flow: 0 (no connection to the sink)
    assert graph.EdmondKarp() == 0, "Edmonds-Karp algorithm failed"
    assert graph.Dinic() == 0, "Dinic's algorithm failed"

    print("Test Case 3 passed")

    # Test Case 4: Large Fully Connected Graph
    capacity_matrix = np.full((5, 5), 10)
    np.fill_diagonal(capacity_matrix, 0)

    # S'assurer que la dernière ligne est remplie de 0 (sink)
    capacity_matrix[-1, :] = 0

    # S'assurer que la première colonne est remplie de 0 (source)
    capacity_matrix[:, 0] = 0
    graph = Graph(capacity_matrix)

    # Expected max flow: Large fully connected graphs should allow high flow values
    expected_flow = sum(capacity_matrix[0])
    assert graph.EdmondKarp() == expected_flow, "Edmonds-Karp algorithm failed"
    assert graph.Dinic() == expected_flow, "Dinic's algorithm failed"

    print("Test Case 4 passed")


if __name__ == "__main__":
    test_graph()
