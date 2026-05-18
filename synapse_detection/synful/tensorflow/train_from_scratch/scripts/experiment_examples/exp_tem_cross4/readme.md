## These scripts are used to train the TEM cross 4 experiment

1. Changes parameters.json for hyperparams

2. Edit path to data in the train.py script.

3. Run generate_network.py

4. Run train.py

# For predict (illustrative example)

Predict always runs from a folder call predict.
How it should look in your disk.
```bash
--scripts
  -- train
    -- your_setup_foldername_during_train # setup_03_tem_cross4
 #this contains all the predict scripts (copy all scripts from an example predict folder to this folder before running). Name should be predict
  -- predict
```
**ALWAYS** run predict from this folder which is inside `scripts`, the synful has internal config flow which require triggering from here.

Now, that your have copied the predict scripts under the predict folder above, imagine your setup name during training is `setup_03_tem_cross4`.
You now need to copy the `ckpt` files and the `config jsons` and `meta` files from this folder to the predict folder. These will indicate which ckpt what config to load during predict. For example, you want to run the prediction with ckpt 300000, you should below files to `predict` folder copy from `setup_03_tem_cross4` :
```bash
test_net.meta
test_net_config.json
train_net.meta
train_net_checkpoint_300000.data-00000-of-00001
train_net_checkpoint_300000.index
train_net_checkpoint_300000.meta
train_net_config.json
```

The predict scripts look for the train folder during inference, hence make sure you replace the `parameters.json` files to point to the name of the correct training folder. 

