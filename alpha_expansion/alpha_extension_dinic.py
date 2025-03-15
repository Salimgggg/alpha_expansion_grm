import numpy as np
import cv2
from ..max_flow.graph import Graph
from ..max_flow.solvers.dinic import DinicSolver


class AlphaExpansion:
    def __init__(self, image_path, source_weight=10, sink_weight=10, sigma=15, source_label=255):
        """
        Segmentation d'image noir et blanc (niveaux de gris) par Graph-Cut avec DinicSolver.
        """
        self.image_path = image_path
        self.source_weight = source_weight
        self.sink_weight = sink_weight
        self.sigma = sigma
        self.source_label = source_label

        # Chargement de l'image en niveaux de gris
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(
                f"Impossible de charger l'image depuis {self.image_path}")

        self.height, self.width = self.image.shape
        # Nombre total de nœuds : pixels + source + sink
        self.size = self.height * self.width + 2
        self.source = self.height * self.width  # Indice du nœud source
        self.sink = self.height * self.width + 1  # Indice du nœud sink

        # Initialisation du graphe (suppose que Graph accepte une taille et peut gérer capacity)
        self.graph = Graph(size=self.size, source=self.source, sink=self.sink)
        self.segmented_image = np.zeros_like(self.image, dtype=np.uint8)

    def neighbor_weight(self, intensity1, intensity2):
        """
        Calcule le poids entre deux pixels voisins selon leur différence d'intensité.
        """
        diff = intensity1 - intensity2
        return np.exp(- (diff ** 2) / (2 * (self.sigma ** 2)))

    def build_graph(self):
        """
        Construit le graphe reliant chaque pixel à ses voisins immédiats.
        """
        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x
                intensity = float(self.image[y, x])

                # Connexion aux voisins immédiats (haut, bas, gauche, droite)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        n_idx = ny * self.width + nx
                        neighbor_intensity = float(self.image[ny, nx])
                        weight = self.neighbor_weight(
                            intensity, neighbor_intensity)
                        # Ajout d'arêtes bidirectionnelles
                        self.graph.capacity[idx, n_idx] = weight
                        self.graph.capacity[n_idx, idx] = weight
        return self.graph

    def add_terminal_nodes(self):
        """
        Ajoute les connexions vers les terminaux : source (objet) et puits (fond).
        """
        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x
                intensity_pixel = float(self.image[y, x])
                intensity_label_current = float(self.segmented_image[y, x])

                # Calcul des coûts
                cost_source = abs(intensity_pixel -
                                  self.source_label) * self.source_weight
                cost_sink = abs(intensity_pixel -
                                intensity_label_current) * self.sink_weight

                # Ajout des arêtes terminales
                self.graph.capacity[self.source,
                                    idx] = cost_sink  # Source -> Pixel
                # Pixel -> Sink
                self.graph.capacity[idx, self.sink] = cost_source
        return self.graph

    def max_flow(self):
        """
        Calcule le flot maximal avec DinicSolver.
        """
        solver = DinicSolver(self.graph)
        self.flow = solver.solve()
        print(f"Flot maximal : {self.flow}")
        self.solver = solver  # Stocke le solveur pour accéder au flow plus tard
        return self.flow

    def segment_nodes(self):
        """
        Construit l'image segmentée finale après calcul du flot maximal.
        """
        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x
                # Si le pixel est connecté à la source (flow[source, idx] > 0), il appartient à l'objet
                if self.solver.flow[self.source, idx] > 0:
                    self.segmented_image[y, x] = self.source_label
        return self.segmented_image


# Exemple d'utilisation
if __name__ == "__main__":
    segmenter = AlphaExpansion("chemin/vers/image.jpg")
    segmenter.build_graph()
    segmenter.add_terminal_nodes()
    segmenter.max_flow()
    result = segmenter.segment_nodes()
    cv2.imwrite("segmentation_result.jpg", result)
