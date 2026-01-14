# Synapse Detection

We support two synapse detection methods, namely, Synful and SimpSyn.

# Synful
Synful can be run using either TensorFlow or Pytorch.
Please follow the instructions under each of their respective folders.

## Quick overview on current results
We have tested Synful in both in-distribution and out-of-distribution settings, such that we can either qualitatively or quantatively assess the effects of domain-drift on the generalisation power of these models.

### In-distribution (ID)
This dataset is an in-house FIBSEM volume of an L1 larva at `8x8x8`, dubbed Octo. Synful MT1 model was both trained and tested on Octo.


<br>
<div>
<p align="center">
<img src='https://github.com/Mohinta2892/catena/blob/dev/synapse_detection/synful/assets/synful_octo_in_distribution_tests.png' align="center" width=800px>
</p>
</div>

### Out-of-distribution (OOD)
Synful models were applied to popular public datasets such as Hemibrain, MANC and WASP in a cross-dataset "validation of generalization" setting. What we mean is that the models were trained on combination of datasets such as Hemibrain, MANC and Octo and then tested on WASP. This gives us an idea of how transferable the learnings are of this model.

<br>
<div>
<p align="center">
<img src='https://github.com/Mohinta2892/catena/blob/dev/synapse_detection/synful/assets/synful_cross_dataset_tests.png' align="center" width=800px>
</p>
</div>

# SimpSyn
SimpSyn is our Simple Synapse Detection algorithm. It is inspired from [Domain Adaptive Synapse Detection with Weak Point Annotations](https://arxiv.org/abs/2308.16461)
but is a Single Stage Synapse Detector. Please see our [Preprint](https://arxiv.org/html/2509.17041v1).

SimpSyn currently leverages [BiaPy](https://github.com/BiaPyX/BiaPy) for training, data loading and inference.
So please follow the installation instructions for BiaPy to run SimpSyn.




