# SIMPSYN
<p align="center">

<img width="942" height="961" alt="simpsyn_logo" src="https://github.com/user-attachments/assets/3a380ad5-307b-4ead-b40d-32c04ba0b0d8" />

</p>


SIMPSYN is our ultra-light weight model for synapse detection.
SimpSyn uses a 3D Residual U-Net [30] for synapse detection that predicts two output channels: one corresponding to pre-synaptic regions and the other to post-synaptic regions. 
To achieve this, a spherical 3D region is generated and centered at the coordinates of the pre-synaptic and post-synaptic points.
These output masks are subsequently processed using connected component labelling to isolate individual synaptic structures.
To establish correspondence between pre-synaptic and post-synaptic sites, each post-synaptic component is paired with its nearest pre-synaptic counterpart based on the nearest neighbour criterion.

<img width="1031" height="550" alt="image" src="https://github.com/user-attachments/assets/68b965c0-0dc0-43c4-a638-7f7d57e60b31" />


# Getting Started
**SIMPSYN is built within Biapy**.
Please follow the [installation instructions here](https://biapy.readthedocs.io/en/latest/get_started/installation.html).

_Complete documentation will be released soon._

# Citing SIMPSYN
Please cite our [preprint on Towards Generalized Synapse Detection Across Invertebrate Species](https://arxiv.org/html/2509.17041v1).

