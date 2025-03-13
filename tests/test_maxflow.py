import numpy as np
from max_flow.graph import Graph
from max_flow.solvers.edmonds_karp import EdmondsKarpSolver
from max_flow.solvers.dinic import DinicSolver
from max_flow.solvers.push_relabel import PushRelabelSolver


def test_graph():
    # Test Case 1: Graphe simple
    capacity_matrix = np.array([
        [0, 10, 10, 0, 0, 0],
        [0, 0, 2, 4, 8, 0],
        [0, 0, 0, 0, 9, 0],
        [0, 0, 0, 0, 0, 10],
        [0, 0, 0, 6, 0, 10],
        [0, 0, 0, 0, 0, 0]
    ])
    graph = Graph(capacity_matrix)
    ek_solver = EdmondsKarpSolver(graph)
    assert ek_solver.solve() == 19, "Edmonds–Karp a échoué"

    graph = Graph(capacity_matrix)
    dinic_solver = DinicSolver(graph)
    assert dinic_solver.solve() == 19, "Dinic a échoué"

    graph = Graph(capacity_matrix)
    pr_solver = PushRelabelSolver(graph)
    assert pr_solver.solve() == 19, "Push–Relabel a échoué"

    print("Test Case 1 passé")

    # Test Case 2: Graphe avec un seul chemin
    capacity_matrix = np.array([
        [0, 5, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4],
        [0, 0, 0, 0]
    ])
    graph = Graph(capacity_matrix)
    ek_solver = EdmondsKarpSolver(graph)
    assert ek_solver.solve() == 3, "Edmonds–Karp a échoué"

    graph = Graph(capacity_matrix)
    dinic_solver = DinicSolver(graph)
    assert dinic_solver.solve() == 3, "Dinic a échoué"

    print("Test Case 2 passé")

    # Test Case 3: Graphe déconnecté (aucun flot)
    capacity_matrix = np.zeros((4, 4), dtype=np.int64)
    graph = Graph(capacity_matrix)
    ek_solver = EdmondsKarpSolver(graph)
    assert ek_solver.solve() == 0, "Edmonds–Karp a échoué"

    graph = Graph(capacity_matrix)
    dinic_solver = DinicSolver(graph)
    assert dinic_solver.solve() == 0, "Dinic a échoué"

    print("Test Case 3 passé")

    # Test Case 4: Graphe complet de petite taille
    capacity_matrix = np.full((5, 5), 10, dtype=np.int64)
    np.fill_diagonal(capacity_matrix, 0)
    capacity_matrix[-1, :] = 0  # Le puits n'a pas d'arêtes sortantes
    capacity_matrix[:, 0] = 0   # La source n'a pas d'arêtes entrantes
    graph = Graph(capacity_matrix)
    ek_solver = EdmondsKarpSolver(graph)
    expected_flow = sum(capacity_matrix[0])
    assert ek_solver.solve() == expected_flow, "Edmonds–Karp a échoué"

    graph = Graph(capacity_matrix)
    dinic_solver = DinicSolver(graph)
    assert dinic_solver.solve() == expected_flow, "Dinic a échoué"

    print("Test Case 4 passé")


if __name__ == "__main__":
    test_graph()
