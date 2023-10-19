import os
import hydra


def get_dataset_folder_name(dataset, test_dataset):
    if dataset == "wsj0":
        if test_dataset == "default":
            dataset_folder_name = "default_spk_ind"
        elif test_dataset == "more-speaker":
            dataset_folder_name = "more_spk_ind"
    elif dataset == "vcc" and test_dataset == "default":
        dataset_folder_name = "default_spk_dep"
    else:
        print("Wrong dataset or test dataset!")

    return dataset_folder_name


def get_root():
    # print(f'Current working directory: {os.getcwd()}')
    CURRENT_ROOT = os.getcwd()
    TMP_ROOT = hydra.utils.get_original_cwd()
    os.chdir(TMP_ROOT)
    # os.chdir('..')
    PROJECT_ROOT = os.getcwd()

    return CURRENT_ROOT, PROJECT_ROOT
