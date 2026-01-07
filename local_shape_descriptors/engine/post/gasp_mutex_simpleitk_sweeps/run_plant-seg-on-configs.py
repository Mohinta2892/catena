import subprocess
import os
from glob import glob
import argparse
import pyinputplus as pyip
import inquirer


def access_configs(configpath, algo):
    config_per_algo = glob(os.path.join(configpath, algo, "*.yaml"))
    return config_per_algo


def run_ws_algo(args, algo_choices):
    configpath = args.configpath
    config_dict = {}
    for algo in algo_choices.values():
        config_dict[algo] = access_configs(configpath, algo)

    for k in config_dict.keys():
        for y in config_dict[k]:
            subprocess.run(["plantseg", "--config", y])

            print()


def main():
    parser = argparse.ArgumentParser(
        prog="run-segmentation-with-plantseg",
        description="Runs Plant-Seg's segmentation algorithms on a input hdf of probability maps",
    )
    parser.add_argument('-c', '--configpath',
                        default="/mnt/wwn-0x5000c500e0dbd55e/plant-seg/segmentation-runs/hemi-brain/data/aff/configs",
                        help='Path to folder containing config files')

    args = parser.parse_args()

    # algo_choices = {'a': 'DtWatershed', 'b': 'MultiCut', 'c': 'GASP', 'd': 'MutexWS', 'e': 'SimpleITK'}
    algo_choices = {
        # 'a': 'DtWatershed',
        #             'b': 'MultiCut',
        'c': 'GASP',
        'd': 'MutexWS',
        'e': 'SimpleITK',
        'f': 'Waterz'
    }

    print(f"You have option to run: \n"
          f"{list(algo_choices.values())}")

    # run_all = pyip.inputYesNo("Do you wish to run all watershed algorithms on the data? y/n")
    run_ws = [
        inquirer.Checkbox('watersheds',
                          message="Please select which would you like to run?",
                          choices=list(algo_choices.values()),
                          ),
    ]
    answers = inquirer.prompt(run_ws)
    # if str(run_all).lower() == 'no':
    #     one_algo = pyip.inputYesNo("Do you wish to run one algorithm on the data? y/n")
    #     if str(one_algo).lower() == 'yes':
    #         algo = pyip.inputMenu(['DtWatershed', 'Multicut', 'GASP', 'MutexWS', 'SimpleITK'], lettered=True,
    #                               numbered=False)
    #
    #     elif str(one_algo).lower() == 'no':
    #         list_algos = [algo_choices[item] for item in input(f"Enter keys to choose from {algo_choices}"
    #                                                            f" you want to run : ").split()]
    # elif str(run_all).lower() == 'yes':
    run_ws_algo(args, algo_choices)
    print()


if __name__ == '__main__':
    main()
