### Installation

- Install (Anaconda)[https://docs.anaconda.com/free/anaconda/install/index.html]/(Miniconda)[https://docs.anaconda.com/free/miniconda/miniconda-install/]
- Make a conda env by running (use the `environment_napari.yml` file):
  ```bash
  $ conda env create -f environment_napari.yml -n napari-env # this will install the particular napari version as mentioned in the yml
  ```

Please install [Napari](https://napari.org/stable/tutorials/fundamentals/installation)

  <details>
      <summary>Verify Napari is installed correctly</summary>
      <br>
      The following should load the napari GUI:
    
      $ conda activate napari-env
      $ napari
  </details>


### Usage

Once `Napari` is installed successfully, you can run `visualise_napari.py`:

> **OPTION 1**:
- Open the script in an editor.
- Edit the shebang to point to your napari-env in `Line 1: #!/home/samia/anaconda3/envs/napari-env/bin/python`.
- Run small datasets (NB: *ONLY* Do this if your dataset is small, script loads all datasets into napari GUI & RAM):
  ```bash
    $ catena/visualize/visualise_napari.py -f /path/to/zarr 
  ```
- Run large datasets via slicing (Preselect select a ROI in the data to visualise):
  - 3D
  ```bash
    $ catena/visualize/visualise_napari.py -f /path/to/zarr -s z1:z2,y1:y2,x1:x2 #3D
  ```
  - 2D
  ```bash
    $ catena/visualize/visualise_napari.py -f /path/to/zarr -sf 60 -st 150 #2D slices 60 to 150 across all datasets
  ```
> **OPTION 2**:
- Run small datasets (NB: *ONLY* Do this if your dataset is small, script loads all datasets into napari GUI & RAM):
  ```bash
    $ conda activate napari-env # call the script with the python which has napari
    $ python catena/visualize/visualise_napari.py -f /path/to/zarr 
  ```
- Run large datasets via slicing (Preselect select a ROI in the data to visualise):
  - 3D
  ```bash
    $ conda activate napari-env
    $ cd catena/visualize/visualise_napari.py -f /path/to/zarr -s z1:z2,y1:y2,x1:x2 #3D
  ```
  - 2D
  ```bash
   $ conda activate napari-env
   $ cd catena/visualize/visualise_napari.py -f /path/to/zarr -sf 60 -st 150 #2D slices 60 to 150 across all datasets
  ```
