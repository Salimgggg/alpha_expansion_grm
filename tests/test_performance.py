import numpy as np
import time
import multiprocessing
import maxflow  # pymaxflow
from max_flow.graph import Graph
from max_flow.solvers.edmonds_karp import EdmondsKarpSolver
from max_flow.solvers.dinic import DinicSolver
from max_flow.solvers.push_relabel import PushRelabelSolver


def create_dense_graph_64x64():
    num_pixels = 64 * 64  # 4,096 nodes for pixels
    num_nodes = num_pixels + 2  # +2 for source (0) and sink (last node)
    capacity_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int64)

    source = 0
    sink = num_nodes - 1
    half_nodes = num_pixels // 2

    for i in range(1, half_nodes + 1):
        capacity_matrix[source, i] = np.random.randint(1, 10)

    for i in range(half_nodes + 1, num_pixels + 1):
        capacity_matrix[i, sink] = np.random.randint(1, 10)

    for i in range(1, half_nodes + 1):
        num_connections = np.random.randint(25, 50)
        targets = np.random.choice(
            range(half_nodes + 1, num_pixels + 1), num_connections, replace=False)
        for j in targets:
            capacity_matrix[i, j] = np.random.randint(1, 5)

    return Graph(capacity_matrix, source, sink)


def create_pymaxflow_graph(capacity_matrix):
    """Convertit une matrice de capacité en graphe pymaxflow."""
    num_nodes = capacity_matrix.shape[0]
    source = 0
    sink = num_nodes - 1

    graph = maxflow.Graph[float](num_nodes, int(capacity_matrix.size // 2))
    nodes = graph.add_nodes(num_nodes)

    for i in range(num_nodes):
        if capacity_matrix[source, i] > 0:
            graph.add_tedge(i, capacity_matrix[source, i], 0)
        if capacity_matrix[i, sink] > 0:
            graph.add_tedge(i, 0, capacity_matrix[i, sink])

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if capacity_matrix[i, j] > 0:
                graph.add_edge(
                    i, j, capacity_matrix[i, j], capacity_matrix[j, i])

    return graph


def worker(solver_class, graph, result_dict):
    """Fonction exécutée dans un processus séparé pour les solveurs personnalisés."""
    # Passer source et sink à la nouvelle instance de Graph
    graph_copy = Graph(graph.capacity, graph.source, graph.sink)
    solver = solver_class(graph_copy)
    start_time = time.time()
    try:
        max_flow = solver.solve()
        end_time = time.time()
        result_dict["max_flow"] = max_flow
        result_dict["time"] = end_time - start_time
    except Exception as e:
        result_dict["error"] = str(e)


def worker_pymaxflow(graph, result_dict):
    """Fonction pour exécuter pymaxflow dans un processus séparé."""
    graph_copy = create_pymaxflow_graph(graph.capacity)
    start_time = time.time()
    try:
        max_flow = graph_copy.maxflow()
        end_time = time.time()
        result_dict["max_flow"] = max_flow
        result_dict["time"] = end_time - start_time
    except Exception as e:
        result_dict["error"] = str(e)


def solve_with_timeout(solver_class, graph, timeout=60, is_pymaxflow=False):
    """Exécute un solveur avec une contrainte de temps."""
    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    if is_pymaxflow:
        process = multiprocessing.Process(
            target=worker_pymaxflow, args=(graph, result_dict))
    else:
        process = multiprocessing.Process(
            target=worker, args=(solver_class, graph, result_dict))

    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(
            f"{solver_class.__name__ if not is_pymaxflow else 'PyMaxFlow'} exceeded {timeout} seconds.")

    if "error" in result_dict:
        raise RuntimeError(
            f"{solver_class.__name__ if not is_pymaxflow else 'PyMaxFlow'} failed: {result_dict['error']}")

    return result_dict["max_flow"], result_dict["time"]


def test_solver_speed_dense_graph():
    graph = create_dense_graph_64x64()
    solvers = {
        "Edmonds-Karp": (EdmondsKarpSolver, False),
        "Dinic": (DinicSolver, False),
        "Push-Relabel": (PushRelabelSolver, False),
        "PyMaxFlow": (None, True)
    }

    results = {}
    for name, (SolverClass, is_pymaxflow) in solvers.items():
        try:
            max_flow, duration = solve_with_timeout(
                SolverClass, graph, timeout=60, is_pymaxflow=is_pymaxflow)
            results[name] = {"max_flow": max_flow, "time": duration}
        except TimeoutError as e:
            print(f"Timeout: {e}")
            results[name] = {"max_flow": None, "time": None}
        except RuntimeError as e:
            print(f"Error: {e}")
            results[name] = {"max_flow": None, "time": None}

    valid_flows = [result["max_flow"]
                   for result in results.values() if result["max_flow"] is not None]
    if valid_flows:
        assert all(flow == valid_flows[0]
                   for flow in valid_flows), "Solvers do not give the same max flow"

    print("\nPerformance Summary:")
    for name, result in results.items():
        if result["max_flow"] is None:
            print(f"{name}: FAILED (Timeout or Error)")
        else:
            print(
                f"{name}: Flow = {result['max_flow']}, Time = {result['time']:.4f} s")

    if "Dinic" in results and "PyMaxFlow" in results and results["Dinic"]["max_flow"] is not None and results["PyMaxFlow"]["max_flow"] is not None:
        if results["Dinic"]["max_flow"] == results["PyMaxFlow"]["max_flow"]:
            print("\n✅ Dinic et PyMaxFlow donnent le même résultat.")
        else:
            print("\n❌ Dinic et PyMaxFlow donnent des résultats différents.")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Nécessaire pour Windows
    test_solver_speed_dense_graph()
