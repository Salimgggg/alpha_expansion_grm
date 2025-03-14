import numpy as np
from max_flow.graph import Graph
from max_flow.solvers.edmonds_karp import EdmondsKarpSolver
from max_flow.solvers.dinic import DinicSolver
from max_flow.solvers.push_relabel import PushRelabelSolver
import pytest


def test_case_1_simple_graph():
    capacity_matrix = np.array([
        [0, 10, 10, 0, 0, 0],
        [0, 0, 2, 4, 8, 0],
        [0, 0, 0, 0, 9, 0],
        [0, 0, 0, 0, 0, 10],
        [0, 0, 0, 6, 0, 10],
        [0, 0, 0, 0, 0, 0]
    ])
    expected_flow = 19

    graph = Graph(capacity_matrix)
    ek_solver = EdmondsKarpSolver(graph)
    assert ek_solver.solve() == expected_flow, "Edmonds-Karp failed on case 1"

    graph = Graph(capacity_matrix)
    dinic_solver = DinicSolver(graph)
    assert dinic_solver.solve() == expected_flow, "Dinic failed on case 1"

    graph = Graph(capacity_matrix)
    pr_solver = PushRelabelSolver(graph)
    assert pr_solver.solve() == expected_flow, "Push-Relabel failed on case 1"


def test_case_2_single_path():
    capacity_matrix = np.array([
        [0, 5, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4],
        [0, 0, 0, 0]
    ])
    expected_flow = 3

    graph = Graph(capacity_matrix)
    ek_solver = EdmondsKarpSolver(graph)
    assert ek_solver.solve() == expected_flow, "Edmonds-Karp failed on case 2"

    graph = Graph(capacity_matrix)
    dinic_solver = DinicSolver(graph)
    assert dinic_solver.solve() == expected_flow, "Dinic failed on case 2"


def test_case_3_disconnected():
    capacity_matrix = np.zeros((4, 4), dtype=np.int64)
    expected_flow = 0

    graph = Graph(capacity_matrix)
    ek_solver = EdmondsKarpSolver(graph)
    assert ek_solver.solve() == expected_flow, "Edmonds-Karp failed on case 3"

    graph = Graph(capacity_matrix)
    dinic_solver = DinicSolver(graph)
    assert dinic_solver.solve() == expected_flow, "Dinic failed on case 3"


def test_case_4_complete_small_graph():
    capacity_matrix = np.full((5, 5), 10, dtype=np.int64)
    np.fill_diagonal(capacity_matrix, 0)
    capacity_matrix[-1, :] = 0  # The sink has no outgoing edges
    capacity_matrix[:, 0] = 0   # The source has no incoming edges
    expected_flow = sum(capacity_matrix[0])

    graph = Graph(capacity_matrix)
    ek_solver = EdmondsKarpSolver(graph)
    assert ek_solver.solve() == expected_flow, "Edmonds-Karp failed on case 4"

    graph = Graph(capacity_matrix)
    dinic_solver = DinicSolver(graph)
    assert dinic_solver.solve() == expected_flow, "Dinic failed on case 4"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
