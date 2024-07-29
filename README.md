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
|  |              EM-to-EM              | |
|  |                                    | |
|  +-----------------+-----------------+  |
+-----------------------------------------+
```



Catena provides a comprehensive workflow for automated connectome reconstruction based on SOTA Funke-lab pipelines for Neuron Segmentation `Local Shape Descriptors (Sheridan et al. 2022)`, Synapse Detection `Synful (Buhmann et al. 2020)`, Microtubule tracking `Micron (Eckstein et al. 2019)` and Neurotransmitter classification `Synister (Eckstein, Bates et al. 2024)` from large-scale volume Electron Microscopy (EM). To cope with the variability across EM datasets, Catena includes popular domain adaptation techniques tailored for EM-to-EM translation.

🛠️ Features:
- Pytorch implementations of `LSDs` and `Synful`.
- Exploration of `LSDs` and `Synful` for other task objectives.
- Docker-based containerisation and release of development environments.
- Style transfer and domain adaptation with Generative AI models.
- Large scale data analysis over public and local EM datasets.
- Artefact logging with Weights and Biases.


***PLEASE NOTE THIS IS UNDER HEAVY DEVELOPMENT. FOLLOW `DEV` BRANCH LINKS BELOW!***

- Neuron Segmentation `Local Shape Descriptors (Sheridan et al. 2022)`: [Installation and Usage ](https://github.com/Mohinta2892/catena/tree/dev/local_shape_descriptors)
- Synapse Detection `Synful(Buhmann et al. 2020)`: [Installation and Usage](https://github.com/Mohinta2892/catena/tree/dev/synful)
- Microtubule tracking `Micron (Eckstein et al. 2019)`: [Installation and Usage ](https://github.com/Mohinta2892/micron-repackaging)
  > [!WARNING] TENSORFLOW 1.x and Gurobi dependencies for ILP
- Neurotransmitter classification `Synister (Eckstein, Bates et al. 2024)`: TO BE ADDED
- Generative AI for EM-to-EM translation: TO BE ADDED

- For visualisation: [Napari and Neuroglancer](https://github.com/Mohinta2892/catena/tree/dev/visualize)


Please check `Issues` for basic troubleshooting tips. Kindly note these packages are being tested gradually and not all issues have made it to the list yet.

## References
The pipeline has been built upon pre-existing work:
- Local Shape Descriptors: [Github](https://github.com/funkelab/lsd), [Paper](https://www.nature.com/articles/s41592-022-01711-z)
- Synful: [GitHub](https://github.com/funkelab/synful), [Paper](https://www.nature.com/articles/s41592-021-01183-7)
- Micron: [Github](https://github.com/nilsec/micron), [Paper](https://arxiv.org/abs/2009.08371)
- Synister: [GitHub](https://github.com/funkelab/synister), [Paper](https://www.cell.com/cell/fulltext/S0092-8674(24)00307-6)
- Generative AI: To do

## Citations
If you use this codebase, please cite us. However, please do not forget to cite the original authors of the algorithms/models.
```
@software{Mohinta_Catena_Neuron_Segmentation_2022,
author = {Mohinta, Samia},
month = aug,
title = {{Catena: Neuron Segmentation, Synapse Detection, Microtubule tracking and more...}},
version = {0.1},
year = {2022}
}
```

## Funding
This work has been supported by generous funding from:

<br>
<div>
<p align="left">
<img src='https://github.com/Mohinta2892/catena/blob/dev/assets/wellcome-logo-black.jpg' align="center" width=100px>
<img src='https://github.com/Mohinta2892/catena/blob/dev/assets/OSSIJanelia_logo.png' align="center" width=200px>
<img src='https://github.com/Mohinta2892/catena/blob/dev/assets/Colour%20logo%20RGB_DM.jpg' align="center" width=200px height=80px>
  
</p>
</div>
<br>

- Symons MCR Conference Fund
- Hugh Paton - JP Morgan Bursaries 
- Dr Teresa Tiffert Research Innovation Awards

### 💥 Research Outputs
#### 🤝 Conferences
- Berlin Connectomics 2024, MPI Berlin, Germany - invited for Poster Presentation
- UK Neural Computation 2024, Sheffield University, Sheffield UK - invited for [Poster Presentation](https://www.dropbox.com/scl/fi/8ei8ff1ygqbym5mcvi47n/PosterJuly_UKNeuralComp2024_UCLNeuroAI.zip?rlkey=annh1n5sbxhy0h5o29fydtzq9&dl=0)
- UCL NeuroAI 2024, UCL, London UK - invited for Poster Presentation


