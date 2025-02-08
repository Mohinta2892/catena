## EM MASK GENERATION

Raw EM data often is surrounding by resin and unnecessary bits of information (e.g., trachea in the **Drosophila**) which can be easily masked out before running inference on the whole CNS volume. This generally speeds up the inference, as "not to be segmented" regions in the volume are automatically
skipped. We have developed a machine learning model that enables us to generate these masks very efficiently on a lower resolution EM volume, which can then be used during inference with LSDs.

### Features

### System Requirements
- GPU: 12GB Nvidia CUDA (Developed on RTX3090)
- RAM: 64GB Minimum
- OS: Linux (Rhel, developed on Ubuntu)
- Conda: For virtual python environments

  Please make a new conda env based on the `environment.yaml` shared. Note that you may have installation issues due to the `lib` files, which are specific to the machine it has been developed on. Get rid of them and try again.
  
### Training

### Inference
