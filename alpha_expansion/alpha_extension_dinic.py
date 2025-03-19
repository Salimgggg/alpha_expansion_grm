import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.abspath("../"))  # Ajoute le dossier parent

from max_flow.graph import Graph
from max_flow.solvers.dinic import DinicSolver

import numpy as np
import cv2
from collections import deque

class AlphaExpansionDinic:
    def __init__(self, image_path, source_weight=10, sink_weight=10, sigma=15, source_label=255):
        """
        Segmentation par Graph-Cut utilisant votre solver Dinic.
        On construit un graphe avec deux nœuds terminaux (source et puits)
        et les pixels intermédiaires.
        """
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Impossible de charger l'image depuis {image_path}")
        self.image = cv2.resize(self.image, (64,64))

        self.height, self.width = self.image.shape
        self.num_pixels = self.height * self.width
        # On ajoute 2 nœuds : source (indice 0) et puits (dernier indice)
        self.N = self.num_pixels + 2  
        self.source = 0
        self.sink = self.N - 1
        self.sigma = sigma
        self.source_weight = source_weight
        self.sink_weight = sink_weight
        self.source_label = source_label
        # Matrice de capacité initialisée à zéro (taille N x N)
        self.capacity = np.zeros((self.N, self.N), dtype=np.int64)
        self.segmented_image = np.zeros_like(self.image, dtype=np.uint8)

    def neighbor_weight(self, intensity1, intensity2):
        """
        Calcule le poids entre deux pixels voisins en fonction de leur différence d’intensité.
        """
        diff = intensity1 - intensity2
        return np.exp(- (diff ** 2) / (2 * (self.sigma ** 2)))

    def build_capacity_matrix(self):
        """
        Construit la matrice des capacités.
        Les indices 1 à N-2 représentent les pixels.
        Pour chaque pixel, on connecte :
          - Le pixel aux pixels voisins (arêtes bidirectionnelles).
          - La source vers le pixel et le pixel vers le puits (liens terminaux).
        """
        scale = 1000  # Pour convertir les poids flottants en entiers
        for y in range(self.height):
            for x in range(self.width):
                # L'indice du pixel dans le graphe est décalé de 1 (source=0)
                idx = 1 + y * self.width + x
                intensity = float(self.image[y, x])
                
                # Connexion avec les voisins (haut, bas, gauche, droite)
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        n_idx = 1 + ny * self.width + nx
                        neighbor_intensity = float(self.image[ny, nx])
                        weight = self.neighbor_weight(intensity, neighbor_intensity)
                        # Ajout d'arêtes bidirectionnelles
                        self.capacity[idx, n_idx] = int(weight * scale)
                        self.capacity[n_idx, idx] = int(weight * scale)
                
                # Calcul des coûts pour les terminaux :
                intensity_label_current = float(self.segmented_image[y,x])
                cost_source = int(abs(intensity - self.source_label) * self.source_weight * scale)
                cost_sink = int(abs(intensity - intensity_label_current) * self.sink_weight * scale)
                self.capacity[self.source, idx] = cost_sink
                self.capacity[idx, self.sink] = cost_source

    def compute_max_flow(self):
        """
        Construit la matrice, crée le graphe et calcule le flot maximal en utilisant l'algorithme de Dinic.
        """
        self.build_capacity_matrix()
        graph = Graph(self.capacity)
        self.solver = DinicSolver(graph)
        max_flow = self.solver.solve()
        print("Flot maximal :", max_flow)
        return max_flow

    def segment_nodes(self):
        """
        Après le calcul du flot maximal, effectue une DFS sur le graphe résiduel
        à partir de la source pour déterminer les pixels accessibles (appartenant à l'objet).
        """
        visited = [False] * self.solver.graph.size
        stack = [self.solver.graph.source]
        while stack:
            u = stack.pop()
            if not visited[u]:
                visited[u] = True
                for v in range(self.solver.graph.size):
                    # Si la capacité résiduelle est positive, alors v est accessible depuis u
                    if not visited[v] and self.solver.graph.capacity[u, v] - self.solver.flow[u, v] > 0:
                        stack.append(v)
        
        # Les pixels sont aux indices de 1 à N-2
        for y in range(self.height):
            for x in range(self.width):
                idx = 1 + y * self.width + x
                if visited[idx]:
                    self.segmented_image[y, x] = self.source_label
                    #print(self.segmented_image)

        return self.segmented_image
