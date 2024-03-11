### Installation

Please install (Napari)[https://napari.org/stable/tutorials/fundamentals/installation]

  <details>
      <summary>Verify Napari is installed correctly</summary>
      <br>
      The following should load the napari GUI:
    
      $ conda activate napari-env
      $ napari
  </details>


### Usage

Once `Napari` is installed successfully, you can run `visualise_napari.py`:

[OPTION 1]:
- Open the script in an editor.
- Edit the shebang to point to your napari-env in `Line 1`.
- Run small datasets (NB: *ONLY* Do this if your dataset is small, script loads all datasets into napari GUI & RAM):
  ```bash
    $ cd catena/visualize/visualise_napari.py -f /path/to/zarr 
  ```
- Run large datasets via slicing (Preselect select a ROI in the data to visualise):

  ```bash
    $ cd catena/visualize/visualise_napari.py -f /path/to/zarr -s z1:z2,y1:y2,x1:x2 #3D
  ```

  ```bash
    $ cd catena/visualize/visualise_napari.py -f /path/to/zarr -sf 60 -st 150 #2D slices 60 to 150 across all datasets
  ```
