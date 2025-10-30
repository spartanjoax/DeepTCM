# Deep learning models for tool condition monitoring
![versions](https://upload.wikimedia.org/wikipedia/commons/6/62/Blue_Python_3.9%2B_Shield_Badge.svg)

This repository contains the Python code used to train, evaluate, and compare **expert-defined** and **AutoML-generated** deep learning models for **tool condition monitoring (TCM)** in **face milling**, as presented in **Chapter 4** of the PhD thesis:

> Peralta Abadía, J.J. *Enhancing Smart Monitoring in Face Milling with Deep Continual Learning*. Mondragon Unibertsitatea, 2025.

---

## Acknowledgements
This work was developed at the [Software and systems engineering](https://www.mondragon.edu/en/research-transfer/engineering-technology/research-and-transfer-groups/software-systems-engineering) and the [High-performance machining](https://www.mondragon.edu/en/research-transfer/engineering-technology/research-and-transfer-groups/high-performance-machining) groups at Mondragon University, as part of the [Digital Manufacturing and Design Training Network](https://dimanditn.eu/es/home).

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 814078 (DIMAND) and by the Department of Education, Universities and Research of the Basque Government under the projects Ikerketa Taldeak (Grupo de Ingeniería de Software y Sistemas IT1519-22 and Grupo de investigación de Mecanizado de Alto Rendimiento IT1443-22).

---

## Overview

This repository implements two complementary approaches for tool flank wear (VB) prediction using deep learning:

1. **Expert-defined models**
   - Custom architectures leveraging domain knowledge (e.g., Meta-learning Ensembles and Robust ResNet).
   - Designed and optimized manually following state-of-the-art deep learning methods for regression in TCM.
   - Described in:
     - Peralta Abadía et al., *Procedia CIRP*, 2024  
     - Peralta Abadía et al., *DYNA*, 2024  

2. **AutoML-generated models**
   - Automatically searched and optimized architectures using [AutoKeras](https://autokeras.com/).
   - Incorporates multimodal learning by combining signal-based and process-information networks.
   - Demonstrates the feasibility of AutoML in scalable TCM model design.

Both model types were trained and benchmarked using the **NASA Ames Milling Dataset**.

---

## ⚙️ Repository Structure
```
AutoML-vs-Expert-TCM/
├── data/
│ ├── TBD # Raw and preprocessed NASA dataset
│ └── TBD/ # Scripts for signal segmentation and normalization
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Getting started

### Prerequisites

- Python ≥ 3.9
- TensorFlow ≥ 2.9
- Keras ≥ 2.9
- AutoKeras == 1.0.20
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Installation
```bash

```bash
pip install -r requirements.txt
```
4. Follow the instructions in the [Usage](#usage) section.

---

## Usage
```bash
# Expert-defined models
python train_expert_models.py

# AutoML search and retraining
python run_automl_search.py
```

---

## Dataset
NASA Ames/UC Berkeley face-milling open-access dataset

Source: [NASA Ames Prognostics Data Repository](https://data.nasa.gov/dataset/milling-wear)

---

# Contributors

[//]: contributor-faces

<a href="https://github.com/spartanjoax"><img src="https://avatars.githubusercontent.com/u/29443664?v=4" title="José Joaquín Peralta Abadía" width="80" height="80"></a>

[//]: contributor-faces

---

## Cite
If you use this repository or its associated datasets, please cite the following works:

```bibtex
@article{Peralta2024a,
  author    = {José{-}Joaquín Peralta Abadía and Mikel Cuesta Zabaljauregui and Félix Larrinaga Barrenechea},
  title     = {A meta-learning strategy based on deep ensemble learning for tool condition monitoring of machining processes},
  journal   = {Procedia CIRP},
  volume    = {126},
  pages     = {429--434},
  year      = {2024},
  doi       = {10.1016/j.procir.2024.08.391}
}

@article{Peralta2024b,
  author    = {José{-}Joaquín Peralta Abadía and Mikel Cuesta Zabaljauregui and Félix Larrinaga Barrenechea},
  title     = {Tool condition monitoring in machining using robust residual neural networks},
  journal   = {DYNA},
  volume    = {99},
  number    = {5},
  year      = {2024}
}

@dataset{Peralta2025a,
  author    = {José{-}Joaquín Peralta Abadía and Mikel Cuesta Zabaljauregui and Félix Larrinaga Barrenechea},
  title     = {MU-TCM face-milling dataset},
  year      = {2025},
  howpublished = {\url{https://hdl.handle.net/20.500.11984/6926}}
}
```