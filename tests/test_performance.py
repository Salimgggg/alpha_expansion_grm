import numpy as np
import time
import multiprocessing
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
        num_connections = np.random.randint(25, 50)  # Ajusté pour 64x64
        targets = np.random.choice(
            range(half_nodes + 1, num_pixels + 1), num_connections, replace=False)
        for j in targets:
            capacity_matrix[i, j] = np.random.randint(1, 5)

    return Graph(capacity_matrix)


def worker(solver_class, graph, result_dict):
    """Function to be run in a separate process."""
    graph_copy = Graph(graph.capacity)
    solver = solver_class(graph_copy)
    start_time = time.time()
    try:
        max_flow = solver.solve()
        end_time = time.time()
        result_dict["max_flow"] = max_flow
        result_dict["time"] = end_time - start_time
    except Exception as e:
        result_dict["error"] = str(e)


def solve_with_timeout(solver_class, graph, timeout=60):
    """Runs a solver with a timeout constraint."""
    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    process = multiprocessing.Process(
        target=worker, args=(solver_class, graph, result_dict))
    process.start()
    process.join(timeout)  # Attendre jusqu'à "timeout" secondes

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(
            f"{solver_class.__name__} exceeded {timeout} seconds.")

    if "error" in result_dict:
        raise RuntimeError(
            f"{solver_class.__name__} failed: {result_dict['error']}")

    return result_dict["max_flow"], result_dict["time"]


def test_solver_speed_dense_graph():
    graph = create_dense_graph_64x64()
    solvers = {
        "Edmonds-Karp": EdmondsKarpSolver,
        "Dinic": DinicSolver,
        "Push-Relabel": PushRelabelSolver
    }

    results = {}
    for name, SolverClass in solvers.items():
        try:
            max_flow, duration = solve_with_timeout(
                SolverClass, graph, timeout=180)
            results[name] = {"max_flow": max_flow, "time": duration}
        except TimeoutError as e:
            print(f"Timeout: {e}")
            results[name] = {"max_flow": None, "time": None}

    # Vérifier que tous les solveurs donnent le même flot maximal (sauf ceux en timeout)
    valid_flows = [result["max_flow"]
                   for result in results.values() if result["max_flow"] is not None]
    if valid_flows:
        assert all(flow == valid_flows[0]
                   for flow in valid_flows), "Solvers do not give the same max flow"

    print("\nPerformance Summary:")
    for name, result in results.items():
        if result["max_flow"] is None:
            print(f"{name}: TIMEOUT")
        else:
            print(
                f"{name}: Flow = {result['max_flow']}, Time = {result['time']:.4f} s")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Ajout pour Windows
    test_solver_speed_dense_graph()
