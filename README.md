# **Image Segmentation with Alpha Expansion using Max-Flow Algorithms**  

## **Overview**  
This project focuses on **image segmentation** using **Alpha Expansion**, a powerful optimization method for multi-label segmentation. The segmentation is formulated as an energy minimization problem that leverages **graph cuts** and is solved using **maximum flow algorithms**.  

## **Implemented Max-Flow Algorithms**  
To efficiently solve the min-cut/max-flow problem, the following **max-flow algorithms** have been implemented:  
- **Boykov-Kolmogorov (BK)**  
- **Dinic's Algorithm**  
- **Push-Relabel Algorithm**  
- **Edmonds-Karp (EK) Algorithm**  

Each algorithm offers different trade-offs in terms of time complexity and performance, allowing for comparative analysis in the context of image segmentation.

## **Project Structure**  
```
📂 image_segmentation  
│── 📂 maxflow
│   ├── bk_maxflow.py  # Boykov-Kolmogorov Algorithm  
│   ├── dinic_maxflow.py  # Dinic's Algorithm  
│   ├── push_relabel.py  # Push-Relabel Algorithm  
│   ├── edmonds_karp.py  # Edmonds-Karp Algorithm  
│  
│── 📂 segmentation  
│   ├── alpha_expansion.py  # Alpha Expansion method  
│   ├── graph_construction.py  # Build graph representation of the image  
│  
│── 📂 utils  
│   ├── image_loader.py  # Load and preprocess images  
│   ├── visualization.py  # Display segmented results  
│  
│── main.py  # Run the segmentation pipeline  
│── README.md  # Project documentation  
│── requirements.txt  # Dependencies  
```

## **Installation**  
1. Clone the repository:  
   ```bash
      git clone https://github.com/Salimgggg/alpha_expansion_grm.git
      cd alpha_expansion_grm
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## **Usage**  
Run the **Alpha Expansion segmentation** using one of the max-flow algorithms:  
```bash
python main.py --algorithm bk  # Options: bk, dinic, push_relabel, ek
```

## **Results & Performance Analysis**  
- The segmentation quality and computation time are analyzed for different **max-flow implementations**.  
- The project provides **visualizations** of segmentation results.  
- Comparisons of execution time across algorithms are included.  

## **Future Work**  
- Extend the project to handle **interactive segmentation**.  
- Implement alternative **pairwise energy functions**.  
- Optimize memory usage for large images.  

## **References**  
- Boykov, Y., & Kolmogorov, V. (2004). "An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision".  
- Greig, D., Porteous, B., & Seheult, A. (1989). "Exact Maximum A Posteriori Estimation for Binary Images".  
