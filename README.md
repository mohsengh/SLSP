# SLSP: Sparse Learning and Similarity Preserving

[![IEEE Paper](https://img.shields.io/badge/IEEE-IST%202020-blue)](https://ieeexplore.ieee.org/document/9345884)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FIST50524.2020.9345884-green)](https://doi.org/10.1109/IST50524.2020.9345884)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black)](https://github.com/mohsengh/SLSP)
[![Cite](https://img.shields.io/badge/Cite-BibTeX-orange)](https://doi.org/10.1109/IST50524.2020.9345884)

Official implementation of the IEEE paper:

**Similarity Preserving Unsupervised Feature Selection based on Sparse Learning**

Published in:

**2020 10th International Symposium on Telecommunications (IST 2020), IEEE**

Authors:

* Hadi Zare
* Mohsen Ghassemi Parsa
* Mehdi Ghatee
* Sasan H. Alizadeh

---

## Keywords

Unsupervised Feature Selection, Similarity Preserving Learning, Sparse Learning, Feature Selection, Clustering, Representation Learning, Spectral Analysis, Dimensionality Reduction, Data Mining, Machine Learning.

---

# If You Use This Code, Please Cite the Paper

This repository accompanies the following IEEE publication.

```bibtex
@inproceedings{Zare2020SLSP,
  author={Zare, Hadi and Parsa, Mohsen Ghassemi and Ghatee, Mehdi and Alizadeh, Sasan H.},
  title={Similarity Preserving Unsupervised Feature Selection based on Sparse Learning},
  booktitle={2020 10th International Symposium on Telecommunications (IST)},
  year={2020},
  pages={50--55},
  doi={10.1109/IST50524.2020.9345884}
}
```

DOI:

https://doi.org/10.1109/IST50524.2020.9345884

⭐ If this repository helps your research, please consider starring the repository and citing the paper.

---

# Overview

SLSP is an unsupervised feature selection framework that simultaneously preserves both global and local similarity structures in data while identifying the most informative features.

The proposed method integrates:

* Global similarity preservation
* Local similarity preservation
* Sparse learning
* Clustering-guided feature selection

into a unified optimization framework.

Unlike many conventional feature selection methods that only preserve local neighborhood information, SLSP jointly exploits global and local data structures to improve clustering performance.

---

# Main Contributions

✔ Global similarity preservation using Symmetric Nonnegative Matrix Factorization (SymNMF)

✔ Local similarity preservation through spectral graph analysis

✔ Sparse feature learning using ℓ₂,₁-norm regularization

✔ Joint optimization of clustering and feature selection

✔ Convergence-guaranteed iterative algorithm

✔ Extensive evaluation on benchmark datasets

---

# Why SLSP?

Many existing unsupervised feature selection methods focus on only one aspect of data structure.

| Method | Local Similarity | Global Similarity | Clustering | Joint Learning |
| ------ | ---------------- | ----------------- | ---------- | -------------- |
| MCFS   | ✓                | ✗                 | ✗          | ✗              |
| UDFS   | ✓                | ✗                 | ✗          | ✓              |
| NDFS   | ✓                | ✗                 | ✓          | ✓              |
| SPFS   | ✗                | ✓                 | ✗          | ✓              |
| GLSPFS | ✓                | ✓                 | ✗          | ✓              |
| SLSP   | ✓                | ✓                 | ✓          | ✓              |

SLSP combines all three components within a unified framework.

---

# Experimental Results

The proposed method was evaluated on ten benchmark datasets:

| Dataset | Domain             |
| ------- | ------------------ |
| ALLAML  | Bioinformatics     |
| Colon   | Bioinformatics     |
| GLIOMA  | Bioinformatics     |
| Lung    | Bioinformatics     |
| BA      | Image Analysis     |
| COIL20  | Object Recognition |
| ORL     | Face Recognition   |
| Yale    | Face Recognition   |
| Isolet  | Speech Recognition |
| Madelon | Artificial Dataset |

---

# Performance Highlights

SLSP was compared against:

* LS
* MCFS
* UDFS
* NDFS
* GLSPFS
* SPUFS

Key findings reported in the paper:

* Best clustering accuracy on six benchmark datasets.
* Consistently ranked among the top-performing methods.
* Significant improvements in NMI on ALLAML and Colon datasets.
* Robust performance across biological, image, speech, and artificial datasets.

---

Evaluation metrics:

* Accuracy (ACC)
* Normalized Mutual Information (NMI)

Each experiment was repeated 20 times and the mean and standard deviation were reported.

---

# Applications

SLSP can be applied to:

* Bioinformatics
* Gene Expression Analysis
* Cancer Classification
* Computer Vision
* Face Recognition
* Speech Processing
* Pattern Recognition
* Data Mining
* Representation Learning
* Clustering Preprocessing

---

# Citation

Please cite the following paper if you use this repository:

```bibtex
@inproceedings{Zare2020SLSP,
  author={Zare, Hadi and Parsa, Mohsen Ghassemi and Ghatee, Mehdi and Alizadeh, Sasan H.},
  title={Similarity Preserving Unsupervised Feature Selection based on Sparse Learning},
  booktitle={2020 10th International Symposium on Telecommunications (IST)},
  year={2020},
  pages={50--55},
  doi={10.1109/IST50524.2020.9345884}
}
```

---

# License

This repository is provided for academic and research purposes.

---

# Contact

Mohsen Ghassemi Parsa

Email: mgparsa@ut.ac.ir
GitHub: https://github.com/mohsengh

For questions, suggestions, bug reports, or collaborations, please open an issue.

---

# Support the Project

If you find this work useful:

⭐ Star the repository

📄 Cite the paper

🔄 Share the repository

These actions help increase the visibility and impact of the research.
