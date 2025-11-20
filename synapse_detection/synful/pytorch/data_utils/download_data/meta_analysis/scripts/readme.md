# Utilities

All utility scripts that help to download and convert data into CREMI format for synapse detection.


## Convert WASP data to CREMI format: `convert_wasp_to_CREMI.py`

To test Synful and SimpSyn on WASP datasets, we have to first convert it into the CREMI format.
For us CREMI data format is the standardised format for all synapse prediction models.
The WASP data in CREMI format is currently [here](https://www.dropbox.com/scl/fo/e4c47npukcvrmuu0a5d6h/AOivboXPljDAkXRXHq9WHxA?rlkey=t2b11g1kqca42oy56l59i56nl&st=kovyixyq&dl=0)

## Convert BEE data to CREMI format: `convert_megalopta_bee_to_cremi.py` 

To test Synful and SimpSyn on BEE datasets (credits to Griffin Badalamente and Valentin Gillet in the Heinze lab, Lund University), we have to first convert these volumes into the CREMI format. The shared data resides [here](https://www.dropbox.com/scl/fo/9wdpq2qqgphi7fcn5t07s/AMxDEXAk6XL7lCQMOnQ8IDA?rlkey=g0x4qc6omdiqzge7bkzai1te9&st=tibmj6mk&dl=0).

Please use [this script](https://github.com/Mohinta2892/catena/blob/dev/synapse_detection/synful/pytorch/data_utils/download_data/meta_analysis/scripts/convert_megalopta_bee_to_cremi.py) to convert the data from `zarr` and `json` to `hdf` files in the CREMI format.

## Convert ANT data to CREMI format: `convert_ant_to_cremi_v2.py` 

To test Synful and SimpSyn on ANT datasets, we have to first convert it into CREMI format, which can be done using this [script](https://github.com/Mohinta2892/catena/blob/dev/synapse_detection/synful/pytorch/data_utils/download_data/meta_analysis/scripts/convert_ant_to_cremi_v2.py).
