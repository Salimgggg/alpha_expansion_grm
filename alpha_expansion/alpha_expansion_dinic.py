import cv2
import numpy as np
from max_flow.graph import Graph
from max_flow.solvers.dinic import DinicSolver


class AlphaExpansion:
    def __init__(self, image_path, source_weight=10, sink_weight=10, sigma=15, source_label=255, resize=(128, 128)):
        self.image_path = image_path
        self.source_weight = source_weight
        self.sink_weight = sink_weight
        self.sigma = sigma
        self.source_label = source_label

        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        self.image = cv2.resize(self.image, resize)
        if self.image is None:
            raise ValueError(
                f"Impossible de charger l'image depuis {self.image_path}")

        self.height, self.width = self.image.shape
        self.segmented_image = np.zeros_like(self.image, dtype=np.uint8)

        self.num_pixels = self.height * self.width
        self.source = self.num_pixels  # Source node index
        self.sink = self.num_pixels + 1  # Sink node index

        self.graph = Graph(self.num_pixels + 2, self.source, self.sink)

    def neighbor_weight(self, intensity1, intensity2):
        diff = intensity1 - intensity2
        return np.exp(- (diff ** 2) / (2 * (self.sigma ** 2)))

    def build_graph(self):
        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x
                intensity = float(self.image[y, x])

                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        n_idx = ny * self.width + nx
                        neighbor_intensity = float(self.image[ny, nx])
                        weight = self.neighbor_weight(
                            intensity, neighbor_intensity)
                        self.graph.add_edge(idx, n_idx, weight)
                        self.graph.add_edge(n_idx, idx, weight)

    def add_terminal_nodes(self):
        intensity_ref = np.median(self.image)
        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x
                intensity_pixel = float(self.image[y, x])
                cost_source = abs(intensity_pixel -
                                  self.source_label) * self.source_weight
                cost_sink = abs(intensity_pixel -
                                intensity_ref) * self.sink_weight
                self.graph.add_edge(self.source, idx, cost_sink)
                self.graph.add_edge(idx, self.sink, cost_source)

    def min_cut(self):
        solver = DinicSolver(self.graph)
        solver.solve()

        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x
                if idx in solver.level and solver.level[idx] != -1:
                    self.segmented_image[y, x] = self.source_label
        return self.segmented_image


if __name__ == "__main__":
    segmenter = AlphaExpansion("images/image_001.jpg")
    segmenter.build_graph()
    segmenter.add_terminal_nodes()
    result = segmenter.min_cut()
    cv2.imwrite("segmentation_result.jpg", result)
    print("✅ Image segmentée sauvegardée sous segmentation_result.jpg")
