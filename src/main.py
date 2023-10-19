from Tools.utils import get_root, SendLine, Pycolor
import hydra
import os
import sys


# run experiment with hydra params
@hydra.main(config_name="hydras")
def run_experiment(hydra):

    # get project-root and cd project-root
    # CURRENT_ROOT = os.getcwd()
    _, PROJECT_ROOT = get_root()

    if hydra.params.modeling_method == "None":
        hydra.params.modeling_method = (
            f"As_{hydra.params.retrain_model_type}-Loss_{hydra.params.retrain_loss}"
        )

    hydra.arg.train_data = f"{hydra.const.data_root}train/{hydra.const.dataset}/"
    hydra.arg.teacher_model = (
        f"{hydra.const.model_root}teacher_model/teacher_{hydra.const.dataset}.model"
    )
    # output directories
    hydra.arg.save_model_root = (
        f"{hydra.const.model_root}trained/wsj0/{hydra.params.modeling_method}/"
    )
    hydra.arg.save_output_root = (
        f"./output/{hydra.const.test_mode}/{hydra.const.dataset}/"
    )
    hydra.arg.save_result_root = (
        f"./result/{hydra.const.test_mode}/{hydra.const.dataset}/"
    )

    hydra.arg.test_dataset_name = None
    hydra.arg.test_data = None
    hydra.arg.model_path = None
    hydra.arg.output_path = None
    hydra.arg.test_dataset_folders = None
    hydra.arg.label_dim = None

    if (
        hydra.const.test_mode == "trained"
        and hydra.params.modeling_method == "ChimeraACVAE"
    ):
        hydra.arg.model_path = f"{hydra.arg.save_model_root}/Chimera.model"
    elif (
        hydra.const.test_mode == "trained"
        and hydra.params.retrain_model_type == "AE"
        and hydra.params.retrain_loss == "ISdiv"
    ):
        hydra.arg.model_path = f"{hydra.arg.save_model_root}/27.model"
    elif (
        hydra.const.test_mode == "trained"
        and hydra.params.retrain_model_type == "AE"
        and hydra.params.retrain_loss == "ISdiv_6ch"
    ):
        hydra.arg.model_path = f"{hydra.arg.save_model_root}/5.model"
    elif (
        hydra.const.test_mode == "trained"
        and hydra.params.retrain_model_type == "AE"
        and hydra.params.retrain_loss == "ISdiv_236ch"
    ):
        hydra.arg.model_path = f"{hydra.arg.save_model_root}/5.model"
        # hydra.arg.model_path = f"{hydra.arg.save_model_root}/24.model"
    elif (
        hydra.const.test_mode == "trained"
        and hydra.params.retrain_model_type == "AE"
        and hydra.params.retrain_loss == "PIT"
    ):
        hydra.arg.model_path = f"{hydra.arg.save_model_root}/21.model"
        # hydra.arg.model_path = f"{hydra.arg.save_model_root}/24.model"

    # define experiment path
    # hydra.arg.test_dataset_name = utils.get_dataset_folder_name(
    #    hydra.const.dataset, hydra.const.test_dataset
    # )
    hydra.arg.test_dataset_name = "jvs"
    hydra.arg.test_data = f"{hydra.const.data_root}test/{hydra.arg.test_dataset_name}/"
    # hydra.arg.test_data = f"{hydra.const.data_root}train/{hydra.arg.test_dataset_name}/"
    hydra.arg.output_path = (
        # f"{hydra.arg.save_output_root}/{hydra.arg.test_dataset_name}/"
        f"{hydra.arg.save_output_root}"
    )

    default_spk_group = [
        "default_spk_dep",
        "default_spk_ind",
    ]
    more_spk_group = ["more_spk_ind", "more_spk_ind_repeat", "jvs"]

    if hydra.arg.test_dataset_name in default_spk_group:
        from constant import DEFAULT_SPK as SPK_DATA
    elif hydra.arg.test_dataset_name in more_spk_group:
        from constant import MORE_SPK as SPK_DATA
    else:
        sys.exit(f"test_dataset:{hydra.arg.test_dataset_name} is None.")
    hydra.arg.test_dataset_folders = SPK_DATA["folders"]
    hydra.arg.label_dim = SPK_DATA["label_dim"]
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # ---------------------------------------------
    # stage 1: test (pre-)trained model
    # ---------------------------------------------
    import source_separation

    if (hydra.const.stage <= 1) and (hydra.const.stop_stage >= 1):
        # run FastMVAE2
        # test_fmvae2.main(hydra, exp_params)
        print("#------------------------------------------------------------------")
        print("# stage 1: Start to separate the mixtures with trained model...    ")
        print("#------------------------------------------------------------------")
        source_separation.main(hydra)
        print(Pycolor.GREEN + "Separation is done." + Pycolor.END)

    # ---------------------------------------------
    # stage 2:  eval mode
    # ---------------------------------------------
    import evaluation

    if (hydra.const.stage <= 2) and (hydra.const.stop_stage >= 2):
        print("#------------------------------------------")
        print("# stage 2: Start to evaluate the outputs...")
        print("#------------------------------------------")
        evaluation.main(hydra)
        print(Pycolor.GREEN + "Evaluation is done." + Pycolor.END)

    # ---------------------------------------------
    # stage 3:  Plotting result
    # ---------------------------------------------
    import show_glaph

    if (hydra.const.stage <= 3) and (hydra.const.stop_stage >= 3):
        print("#----------------------------------------")
        print("# stage 3: Start to plot result glaphs...")
        print("#----------------------------------------")
        show_glaph.main(hydra)
        print(Pycolor.GREEN + "Plotting is done." + Pycolor.END)

if __name__ == "__main__":
    run_experiment()