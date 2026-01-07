import os.path

import numpy as np
import yaml
import itertools
import timeit


def read_yaml(file_path):
    with open(file_path, "r") as stream:
        try:
            file_y = yaml.safe_load(stream)
            # print(file_y)
        except yaml.YAMLError as exc:
            print(exc)

        return file_y


def edit(file_y, file_path, probmap_path=None):
    # {'path': None,
    # 'preprocessing': {'state': False, 'save_directory': 'PreProcessing', 'factor': [1.0, 1.0, 1.0], 'order': 2,
    #                   'crop_volume': '[:,:,:]', 'filter': {'state': False, 'type': 'gaussian', 'filter_param': 1.0}},
    # 'cnn_prediction': {'state': False, 'model_name': 'generic_confocal_3d_unet', 'device': 'cuda',
    #                    'mirror_padding': [16, 32, 32], 'num_workers': 8, 'patch': [32, 128, 128],
    #                    'stride': [20, 100, 100], 'version': 'best', 'model_update': False},
    # 'cnn_postprocessing': {'state': False, 'tiff': False, 'factor': [1, 1, 1], 'order': 2},
    # 'segmentation': {'state': True, 'name': 'MultiCut', 'beta': 0.5, 'save_directory': 'MultiCut', 'run_ws': True,
    #                  'ws_2D': True, 'ws_threshold': 0.5, 'ws_minsize': 50, 'ws_sigma': 2.0, 'ws_w_sigma': 0,
    #                  'post_minsize': 50},
    # 'segmentation_postprocessing': {'state': True, 'tiff': True, 'factor': [1, 1, 1], 'order': 0, 'save_raw': False}}

    algo_names = ['MultiCut', 'GASP', 'MutexWS', 'DtWatershed', 'SimpleITK']
    betas = np.arange(0.1, 0.9, 0.1)
    ws_2D = False  # False]
    ws_threshold = np.arange(0, 1.1, 0.1)
    ws_minsize = 10  # np.arange(1, 106, 5)
    ws_sigma = [0.2, 2]  # np.arange(0.1, 4.9, 1)
    ws_w_sigma = [0.2, 2]  # np.arange(0, 5.1, 1)
    post_minsize = 200  # np.arange(1, 200, 5)
    function_area = "segmentation"
    file_y[function_area]["post_minsize"] = post_minsize
    file_y[function_area]["ws_minsize"] = ws_minsize
    file_y[function_area]["ws_2D"] = ws_2D
    file_y["segmentation_postprocessing"]["tiff"] = True

    # independent variable
    file_y["path"] = probmap_path

    list_of_lists = []
    list_of_lists.append(algo_names)
    list_of_lists.append(list(betas))
    # list_of_lists.append(ws_2D)
    list_of_lists.append(list(ws_threshold))
    # list_of_lists.append((ws_minsize))
    list_of_lists.append(ws_sigma)
    list_of_lists.append(ws_w_sigma)

    all_combos = list(itertools.product(*list_of_lists))
    keys_for_combos = ["name", "beta", "ws_threshold", "ws_sigma", "ws_w_sigma"]

    for i in range(len(all_combos)):
        try:
            combo = all_combos[i]
            save_directory = combo[0]

            save_directory += '_'.join([x + str(y) for x, y in zip(keys_for_combos[1:], combo[1:])])
            file_y[function_area]["save_directory"] = save_directory

            print(f"Saving here: \n {save_directory}")

            for ci in range(len(combo)):
                try:
                    k = keys_for_combos[ci]
                    v = combo[ci]
                    if isinstance(v, float):
                        v = float(v)

                    file_y[function_area][k] = v

                    outfile = os.path.join(os.path.dirname(probmap_path), 'configs', combo[0])
                    if not os.path.exists(outfile):
                        os.makedirs(outfile)

                    with open(f"{os.path.join(outfile, save_directory)}.yaml", 'w') as file:
                        yaml.dump(file_y, file)

                except Exception as e:
                    print(e)
        except Exception as e:
            print(e)

        print()


def main():
    file_path = "/mnt/wwn-0x5000c500e0dbd55e/plant-seg/plantseg-config.yaml"
    probmap_path = "/mnt/wwn-0x5000c500e0dbd55e/plant-seg/segmentation-runs/hemi-brain/data/aff/pred_roi_1_z0-512_y0-512_x0-512-z250-270_y200-400_x200-400.hdf"

    file_y = read_yaml(file_path=file_path)

    start = timeit.default_timer()

    edit(file_y, file_path, probmap_path=probmap_path)

    stop = timeit.default_timer()

    print('Time: ', stop - start)


if __name__ == "__main__":
    main()
