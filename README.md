# Catena


```bash
+-----------------------------------------+
|                Catena                   |
|                                         |
|  +-----------------+-----------------+  |
|  | Neuron          | Synaptic Pair    | |
|  | Segmentation    | Detection        | |
|  |                 |                  | |
|  +-----------------+-----------------+  |
|  +-----------------+-----------------+  |
|  | Microtubule     | Neurotransmitter | |
|  | Tracking        | Classification   | |
|  |                 |                  | |
|  +-----------------+-----------------+  |
|  +-----------------+-----------------+  |
|  |                                    | |
|  |           Generative AI            | |
|  |                                    | |
|  |                                    | |
|  +-----------------+-----------------+  |
+-----------------------------------------+
```



Catena provides a comprehensive workflow for automated connectome reconstruction based on SOTA Funke-lab pipelines for Neuron Segmentation `Local Shape Descriptors (Sheridan et al. 2022)`, Synapse Detection `Synful (Buhmann et al. 2020)`, Microtubule tracking `Micron (Eckstein et al. 2019)` and Neurotransmitter classification `Synister (Eckstein, Bates et al. 2024)` from large-scale volume Electron Microscopy (EM). To cope with the variability across EM datasets, Catena includes popular domain adaptation techniques tailored for EM-to-EM translation.

🛠️ Features:
- Pytorch implementations of `LSDs` and `Synful`.
- Exploration of `LSDs` and `Synful` for other task objectives.
- Containerisation and release of development environments as Dockers.
- Style transfer and domain adaptation with Generative AI models.
- Large scale data analysis over public and local EM datasets.

***PLEASE NOTE THIS IS UNDER HEAVY DEVELOPMENT. FOLLOW `DEV` BRANCH LINKS BELOW!***

- Neuron Segmentation `Local Shape Descriptors (Sheridan et al. 2022)`: [Installation and Usage ](https://github.com/Mohinta2892/catena/tree/dev/local_shape_descriptors)
- Synapse Detection `Synful(Buhmann et al. 2020)`: [Installation and Usage](https://github.com/Mohinta2892/catena/tree/dev/synful)
- Microtubule tracking `Micron (Eckstein et al. 2019)`: [Installation and Usage ](https://github.com/Mohinta2892/micron-repackaging)
  > [!WARNING] TENSORFLOW 1.x and Gurobi dependencies for ILP
- Neurotransmitter classification `Synister (Eckstein, Bates et al. 2024)`: TO BE ADDED
- Generative AI for EM-to-EM translation: TO BE ADDED

- For visualisation: [Napari and Neuroglancer](https://github.com/Mohinta2892/catena/tree/dev/visualize)


Please check `Issues` for basic troubleshooting tips. Kindly note these packages are being tested gradually and not all issues have made it to the list yet.

### 💥 Research Outputs
#### 🤝 Conferences
- Berlin Connectomics 2024, MPI Berlin, Germany - invited for Poster Presentation
- UK Neural Computation 2024, Sheffield University, Sheffield UK - invited for Poster Presentation
- UCL NeuroAI 2024, UCL, London UK - invited for Poster Presentation


