# CATENA
<div align="center">
  <img width="411" alt="image" src="https://github.com/user-attachments/assets/701de2ef-502e-4397-8adb-8ae365e7ae74" />
</div>

<div align="center">
  Talk to us: <a href="https://discord.gg/KEkmVGXRjn">Discord</a>
</div>


# Overview of CATENA
CATENA provides a comprehensive workflow for automated connectome reconstruction based on SOTA Funke-lab pipelines for Neuron Segmentation `Local Shape Descriptors (Sheridan et al. 2022)`, Synapse Detection `Synful (Buhmann et al. 2020)`, Microtubule tracking `Micron (Eckstein et al. 2019)` and Neurotransmitter classification `Synister (Eckstein, Bates et al. 2024)` from large-scale volume Electron Microscopy (EM). To cope with the variability across EM datasets, Catena includes popular domain adaptation techniques tailored for EM-to-EM translation.

Directory Structure (in its present state) is [here](https://github.com/Mohinta2892/catena/blob/dev/assets/directory_structure.md).

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
- Neurotransmitter classification `Synister (Eckstein, Bates et al. 2024)`: [Installation and Usage](https://github.com/Mohinta2892/catena/tree/dev/neurotransmitter_classification)
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
- Dr Teresa Tiffert Research Innovation Award

## Usage Collaborations
This work is being used in other institutes:

<br>
<div>
<p align="left">
<img src='https://github.com/Mohinta2892/catena/blob/dev/assets/CRICK_Logo1.jpg' align="center" width=100px>
<img src='https://github.com/Mohinta2892/catena/blob/dev/assets/UCL_logo.jpg' align="center" width=200px>
  
</p>
</div>
<br>

## 💬 What People Are Saying About Catena

<table>
<tr>
    <td align="center" valign="top" width="100px">
    <a href="https://www.crick.ac.uk/research/labs/michael-winding">
        <img src="https://github.com/Mohinta2892/catena/blob/dev/assets/CRICK_Logo1.jpg?size=100" alt="crick logo" width="100" />
        <br />
        <b>Winding Lab, The Crick</b>
    </a>
    </td>
    <td valign="top">
    "I was very positively surprised by the quality of the segmentations, especially given that the model had not been trained on our data and that only minimal enhancement was applied to the EM images. Larger spines, in particular, are segmented with incredible precision and the identities of individual neurons appear to be well maintained across the z-plane. I was especially impressed to see the model perform well even on noisier regions with low contrast or staining residue in the intracellular space. There are occasional minor errors around small dendritic spines, so I’m very excited to see how the model performs on a dataset that has not undergone the full suite of preprocessing steps.
      
-- Anna Seggewisse "
    <br /><br />
    <a href="https://x.com/WindingMichael/status/1834641737853096243">View on X →</a>
    </td>
</tr>
</table>

### 💥 Research Outputs
#### 🤝 Conferences
- Berlin Connectomics 2024, MPI Berlin, Germany - accepted for Poster Presentation
- UK Neural Computation 2024, Sheffield University, Sheffield UK - accepted for [Poster Presentation](https://www.dropbox.com/scl/fi/8ei8ff1ygqbym5mcvi47n/PosterJuly_UKNeuralComp2024_UCLNeuroAI.zip?rlkey=annh1n5sbxhy0h5o29fydtzq9&dl=0)
- UCL NeuroAI 2024, UCL, London UK - accepted for Poster Presentation
- AI Revolution Meets 4D Cellular Physiology March 2025, HHMI Janelia, USA - accepted for [Poster Presentation](https://tinyurl.com/4dcp-janelia)
- Analysis and Modelling of Connectomes June 2025, HHMI Janelia, USA - accepted for [Poster Presentations](https://ncr25-hhmi.ipostersessions.com/default.aspx?s=A5-9B-8C-8F-83-90-D0-27-96-1B-E9-B8-85-47-2B-46&guestview=true)


