import numpy as np
import cv2
import maxflow

class AlphaExpansion:
    def __init__(self, image_path, source_weight=10, sink_weight=10, sigma=15, source_label=255):
        """
        Segmentation d'image noir et blanc (niveaux de gris) par Graph-Cut.

        Paramètres :
            image_path    : Chemin vers l'image à segmenter.
            source_weight : Poids de connexion à la source (objet).
            sink_weight   : Poids de connexion au puits (arrière-plan).
            sigma         : Paramètre contrôlant la sensibilité aux différences d'intensité.
            source_label  : Intensité de gris représentant l'objet (par défaut blanc pur 255).
        """
        self.image_path = image_path
        self.source_weight = source_weight
        self.sink_weight = sink_weight
        self.sigma = sigma
        self.source_label = source_label

        # Chargement de l'image en niveaux de gris
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Impossible de charger l'image depuis {self.image_path}")

        self.height, self.width = self.image.shape
        self.graph = maxflow.Graph[float](self.height * self.width, 4 * self.height * self.width)
        self.nodes = self.graph.add_nodes(self.height * self.width)

        # Initialisation de l'image segmentée à zéro (fond noir)
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
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        n_idx = ny * self.width + nx
                        neighbor_intensity = float(self.image[ny, nx])
                        weight = self.neighbor_weight(intensity, neighbor_intensity)
                        # Ajout d'arêtes bidirectionnelles entre pixels voisins
                        self.graph.add_edge(idx, n_idx, weight, weight)
        return self.graph, self.nodes

    def add_terminal_nodes(self):
        """
        Ajoute les connexions vers les terminaux : source (objet) et puits (fond).
        
        On considère ici que les pixels proches du label source_label appartiennent à l'objet,
        et ceux éloignés appartiennent au fond.
        """
        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x
                intensity_pixel = float(self.image[y,x])

                # Intensité actuelle du label dans segmented_image
                intensity_label_current = float(self.segmented_image[y,x])

                # Calcul des coûts selon ta demande :
                cost_source = abs(intensity_pixel - self.source_label) * self.source_weight
                cost_sink   = abs(intensity_pixel - intensity_label_current) * self.sink_weight

                # Ajout des arêtes terminales au graphe 
                self.graph.add_tedge(idx, cost_sink, cost_source)
        return self.graph, self.nodes

    def max_flow(self):
        self.flow = self.graph.maxflow()
        print(f"Flot maximal : {self.flow}")
        return self.flow

    def segment_nodes(self):
        """
        Construit l'image segmentée finale après calcul du flot maximal.
        
        Renvoie :
            Image binaire segmentée (objet en blanc et fond en noir).
        """
        for idx in range(len(self.nodes)):
            y, x = divmod(idx, self.width)
            if self.graph.get_segment(idx) == 1:  # Segment objet/source 
                self.segmented_image[y,x] = self.source_label   
        
        return self.segmented_image
