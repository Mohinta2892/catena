## Running notes
Given downsample factors for iso [2, 2, 3] across every axis.
An input shape that can be divided by these factors is needed. For single task and multi-task models 196^3 works fine for both 2D and 3D.
For Auto-context models, we need to provide bigger input shapes such that the LSD output shapes are around 196^3, if the final affinities shapes are to be 72^3.
