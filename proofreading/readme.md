## This folder will primarily host scripts that may allow us to proofread automatically or in a guided way
Initial dev scripts will be for caveclient.

>[!Note]
> For small ROIs please use [Seg2Link](https://github.com/Mohinta2892/Seg2Link)

CAVE/FlyWire requires Gcloud CLI setup. Follow instructions [here](https://cloud.google.com/sdk/docs/install-sdk)


## Synapse Proofreading via Importing to CATMAID

1. Currently our [evaluation script](https://github.com/Mohinta2892/catena/blob/dev/synful/eval/predictions/synapse_partners_pairwise.py) saves the synaptic pairs as `csvs` to be imported to CATMAID. An extension is in the works to include all predictions, irrespective of eval against ground-truth.
2. Example Script to import predicted synapses into CATMAID is [here](https://github.com/shiyanlee/synapse_CATMAID/blob/push_synapses/code%20/import_to_catmaid/push_connectors.ipynb). You would need the `csvs` from #1.
