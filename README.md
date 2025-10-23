# Graph Neural Networks Powered by Encoder Embedding for Improved Node Learning 

[![View Slides](https://img.shields.io/badge/Slides-PDF-blue)](presentations/slides.pdf)
[![View Poster](https://img.shields.io/badge/Poster-PDF-green)](presentations/poster.pdf)

This repository contains the official source code for the paper: ["Graph Neural Networks Powered by Encoder Embedding for Improved Node Learning"](https://arxiv.org/pdf/2507.11732). 

In this work, we leverage a statistically grounded method, one-hot graph encoder embedding (GEE), to generate high-quality initial node features that enhance the end-to-end training of GNNs. We refer to this integrated framework as the **GEE-powered GNN (GG)**. We further propose a concatenated variant, **GG-C**, which concatenates the outputs of GG and GEE. 

*   **Code Structure:**
    * The `clustering/` directory contains all code and experiments related to the node clustering task.
    
    * The `classification/` directory contains all code and experiments for the semi-supervised node classification task.
    
    * Each of these directories includes scripts for both `simulation` experiments on synthetic graphs and `real_data` experiments on benchmark datasets.

*   **Datasets:**
    
    The real-world datasets used in our experiments are sourced from these excellent repositories:
    *   [cshen6/GraphEmd](https://github.com/cshen6/GraphEmd)
    *   [yueliu1999/Awesome-Deep-Graph-Clustering](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering)

*   **Presentations:**
    * Contributed Talk: Slides from our accepted talk for the Joint Statistical Meetings (JSM) 2025. 
    * Poster Presentation: Poster presented at the JHU Data Science and AI (DSAI) Poster Symposium 2025.
    
    ​	
