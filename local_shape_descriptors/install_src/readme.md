We have an existing issue with installing waterz (which does the watershedding and agglomeration) on the predicted affinities.
So, this a workaround for now, which let's us place a pre-installed directly into a new conda environment to run the code for local shape descriptors.


#### Follow these steps
- Conda create the environment from the `.yml` file.
- ```bash 
  conda activate env_name  --> PUT YOUR ENV_NAME
  ```
  Please ensure `numpy==1.24.4` and `cython==0.29.34` in the `.yml`.
  
- Download the above zip files. Unzip them.
- Find the path to `site-packages` for this conda environment's python with:
```bash 
  pip show wandb
  ``` 
  This should give back a path like ` ~/anaconda3/envs/env_name/lib/python3.10/site-packages`.
  Sanity check that the path exists: 
  ```bash
  cd ~/anaconda3/envs/env_name/lib/python3.10/site-packages --> PUT YOUR PATH
  ```
- Copy  waterz and waterz-dist-info into your python envs site-packages. <br>
 **EDIT the PATHS below**.
```bash
  cp -r ~/Downloads/waterz ~/Downloads/waterz-0.9.5.dist-info ~/anaconda3/envs/env_name/lib/python3.10/site-packages
```

**  Sanity check that pip has now access to waterz the new environment **:
```bash
conda activate env_name
python -m pip show waterz
```

The python should be the new conda environment's python. If you see `WARNING: Package(s) not found: waterz`, first check pip is pointing to the
correct python with `which python`. If it is not, run `whereis python`.
Then do, `~/anaconda3/envs/env_name/bin/python -m pip show waterz` should show that waterz is installed.

Any further issues, please report!!


