### Installation

- Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)/[Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/)
- Make a conda env for NAPARI by running (use the `environment_napari.yml` file):
  ###### this will install the particular napari version as mentioned in the yml
  ```bash
  conda env create -f environment_napari.yml -n napari-env 
  ```
- Make a conda env for NEUROGLANCER by running (use the `environment_neuroglancer.yml` file):
  ###### this will install the particular neuroglander version as mentioned in the yml. You may update neuroglancer, but code execution is not guaranteed.
  ```bash
  conda env create -f environment_neuroglancer.yml -n nglancer 
  ```

> [!WARNING]  
> If you face issues with napari, install from [here](https://napari.org/stable/tutorials/fundamentals/installation).
> Make sure to pip install the other dependencies in `environment_napari.yml`. 

<details close>
    <summary> Verify Napari GUI loads</summary>
    <br>
     
    ```
    
    conda activate napari-env
    napari
    
    ```
</details>

<details close>
    <summary> Verify Neuroglancer loads via import </summary>
    <br>
     
    ```
    
    conda activate nglancer
    python -c "import neuroglancer"
    
    ```
</details>

### Usage: NAPARI

>[!IMPORTANT]
>Note that LSDs output segmentations that are smaller than the original EM shape. This is due to application of valid convolutions in the CNN model. So you must pad the output to match the raw shape. Follow [example here](https://github.com/Mohinta2892/catena/blob/dev/visualize/pad_seg_for_napari_viz.md).

Once `Napari` is installed successfully, you can run `visualise_napari.py`:

> **OPTION 1**:
- Open the script in an editor.
- Edit the shebang to point to your napari-env in `Line 1: #!/home/samia/anaconda3/envs/napari-env/bin/python`.
- Run small datasets (NB: *ONLY* Do this if your dataset is small, script loads all datasets into napari GUI & RAM):
  ```bash
    catena/visualize/visualise_napari.py -f /path/to/zarr 
  ```
- Run large datasets via slicing (Preselect select a ROI in the data to visualise):
  - 3D
  ```bash
    catena/visualize/visualise_napari.py -f /path/to/zarr -s z1:z2,y1:y2,x1:x2 #3D
  ```
  - 2D
  ```bash
    catena/visualize/visualise_napari.py -f /path/to/zarr -sf 60 -st 150 #2D slices 60 to 150 across all datasets
  ```
> **OPTION 2**:
- Run small datasets (NB: *ONLY* Do this if your dataset is small, script loads all datasets into napari GUI & RAM):
  ```bash
    conda activate napari-env # call the script with the python which has napari
    python catena/visualize/visualise_napari.py -f /path/to/zarr 
  ```
- Run large datasets via slicing (Preselect select a ROI in the data to visualise):
  - 3D
  ```bash
    conda activate napari-env
    cd catena/visualize/visualise_napari.py -f /path/to/zarr -s z1:z2,y1:y2,x1:x2 #3D
  ```
  - 2D
  ```bash
   conda activate napari-env
   cd catena/visualize/visualise_napari.py -f /path/to/zarr -sf 60 -st 150 #2D slices 60 to 150 across all datasets
  ```

### Usage: Neuroglancer
> [!WARNING]
> All data is expected to be 3D (or 4D for predicted affinities) per the example script. 

- You must edit `nglancer_pyconnectomics_example.py` to suit your datasets that you want to visualise. **This is currently an example script**. Specially,
```bash
  raw_file: pass your own file with contains raw EM
  scales: change the scales which are set as 8nm in zyx
  names: change the zyx to xyz if your data is in that format
```
- The script expects an affinity dataset and a segmentation. Our raws generally contain `raw`, `pred_affs` and `segmentation` in the same `zarr` file.
You must either comment these lines or provide the files with these datasets to plot them in neuroglancer.



# Script features and functionalities
[nglancer_pyconnectomics_example.py](https://github.com/Mohinta2892/catena/blob/dev/visualize/nglancer_pyconnectomics_example.py)
#### Functionality:
This is an example script that demonstrates how to use neuroglancer to visualize 3D (or 4D for predicted affinities) datasets. It expects an affinity dataset and a segmentation. The script sets up a neuroglancer viewer, defines coordinate spaces, and loads raw data, segmentation, and affinities from a zarr file.

#### Usage:

>[!WARNING]
> You must edit nglancer_pyconnectomics_example.py to suit your datasets that you want to visualise.
> This is currently an example script.

- Specially, you need to edit the following:
```python
  raw_file: pass your own file with contains raw EM
  scales: change the scales which are set as 8nm in zyx
  names: change the zyx to xyz if your data is in that format
```
- To run the script:
```bash
conda activate nglancer
python nglancer_pyconnectomics_example.py
```
The script will print a URL to the console that you can open in your browser to see the visualization.
