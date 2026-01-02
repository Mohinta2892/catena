## Write segmentations in zarr into precomputed format
It may be hard to visualize data stored in zarr. If you can convert your segmentations or raw into the neuroglancer precomputed formats, 
you may be able to load them into a google bucket, and directly point the neuroglancer's demo app to this data.

With tensorstore, the raw EM and the segmentations can also be saved in a much more compressed form on disk.

This folder contains the scripts that enable you to `scout and convert` existing zarr volumes (datasets) into precomputed format.
Both `scout and convert` are essentially the same in logic. Given a good workstation, you could run both in parallel and that is why we have disentangled them.

The scout uses a brain resin mask for the EM at much lower resolution (we use scale 4 or scale 5 of a n5 pyramid; 32x smaller than the scale 0 resolution ) that essentially finds which regions contain the relevant brain tissue
and write them, while automatically skipping and writing the rest as background. This can easily save days of work.
