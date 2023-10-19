import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import make_df
from constant import FIG_SIZZE
import hydra
import glob

import seaborn as sns
import matplotlib


matplotlib.rcParams["lines.linewidth"] = 2
matplotlib.rcParams["lines.markersize"] = 6
matplotlib.rcParams["lines.markeredgewidth"] = 0.0
n_mic = "[2-6]"

ITR_MIN = 0
ITR_MAX = 30
SDR_MIN = -4
SDR_MAX = 12

# for 3ch
OBJ_MAX = -2.8 * 10**6
# OBJ_MIN = -3.0 * 10**6

# for 2ch
# OBJ_MAX = -2.80 * 10**6
OBJ_MIN = -3.20 * 10**6

OBJ_MAX = -2.5 * 10**6
# OBJ_MIN = -3.5 * 10**6

FIG_HEIGHT = 120  # [mm]
FIG_ASPECT = 2 / 1  # [横/縦]
METHODS = {"ILRMA": 0, "ChimeraACVAE": 1, "As_AE-Loss_ISdiv": 2}

plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"


def _calc_rate_increase_obj(tmp_df):
    n_inc, n_dec = 0, 0
    costW_series = tmp_df["CostW"]
    pre_cost_value = tmp_df["CostW"][1]

    for tmp_cost_value in costW_series[2:]:
        if pre_cost_value < tmp_cost_value:  # 目的関数増加（良くない）
            n_inc += 1
        else:  # 目的関数減少（良い）
            n_dec += 1
        # 前itrの目的関数値を更新
        pre_cost_value = tmp_cost_value
    return n_inc, n_dec, costW_series.size - 2, tmp_df["ModelingMethod"][0]


def _pallet_and_order(compare_methods):
    if compare_methods == "all":
        palette = ["#35589A", "#5E3783", "#F14A16"]
        # palette = ["#35589A", "#5E3783", "#F14A16", "#F132F5"]
        hue_order = [
            "ILRMA",
            "FastMVAE",
            "Proposed_use_2ch",
            # "As_AE-Loss_PIT",
            # "Proposed_use_6ch",
            # "Proposed_use_236ch",
        ]

    elif compare_methods == "ILRMA_prop.":
        palette = ["#35589A", "#F14A16"]
        hue_order = ["ILRMA", "Proposed"]

    elif compare_methods == "Chimera_prop.":
        palette = ["#5E3783", "#F14A16"]
        hue_order = ["FastMVAE", "Proposed"]
    return palette, hue_order


def calc_imp(df):

    mix_sdr = df[df["Itration"] == 0]["SI-SDR"].copy()
    df["imp_sdr"] = df["SI-SDR"] - float(mix_sdr)
    # print(df.head())

    return (df,)  # df_sir, df_sar


def return_concat_df(df):
    df1 = df.iloc[:, [6]].copy()
    df1 = df1.rename(columns={"CostR": "tmp_cost"})
    df2 = df.iloc[:, [7]].copy()
    df2 = df2.rename(columns={"CostW": "tmp_cost"})

    df1["i"] = list(range(0, len(df1.index) * 2, 2))  # 偶数
    df2["i"] = list(range(1, len(df2.index) * 2, 2))  # 奇数
    res_df = pd.concat([df1, df2])
    res_df = res_df.sort_values("i").reset_index(drop=True).drop("i", axis=1)
    # a = list(np.arange(0,len(df2.index),0.5))
    res_df["Itration"] = list(np.arange(0, len(df2.index), 0.5))
    # print(res_df)
    res_df["ModelingMethod"] = df["ModelingMethod"][0]
    res_df["n_src"] = df["NumberOfSources"][0]
    return res_df


def res_itr(hydra, df, res_root, id):
    ax = sns.relplot(
        data=df,
        kind="line",
        x="Itration",
        y="imp_sdr",
        row="Number of sources",
        # height=213 * FIG_SIZZE["pt"],
        height=FIG_HEIGHT * FIG_SIZZE["mm"],
        aspect=FIG_ASPECT,
        legend=False,
        style="ModelingMethod",
        # ci='sd',
        ci=None,
        facet_kws={"sharey": True, "sharex": False},
        hue="ModelingMethod",
        palette=_pallet_and_order(hydra.glaph.compare_methods)[0],
        hue_order=_pallet_and_order(hydra.glaph.compare_methods)[1],
        # markers = ['v','*','v','*','o'],
    ).set(
        xlabel="Iteration",
        ylabel=f"{id} improvement [dB]",
        xlim=[ITR_MIN, ITR_MAX],
        ylim=[SDR_MIN, SDR_MAX],
        # title="",
    )
    plt.subplots_adjust(hspace=0.4)  # 図同士の感覚を調整
    plt.legend(labels=_pallet_and_order(hydra.glaph.compare_methods)[1], loc="best")
    plt.savefig(os.path.join(res_root, f"res_itr_{id}.pdf"))


def cost_itr(hydra, df, res_root, id, start, end):
    df = df[(df["Itration"] >= start)]
    df = df[(df["Itration"] <= end)]
    ax = sns.relplot(
        data=df,
        kind="line",
        x="Itration",
        y="CostW",
        row="Number of sources",
        height=FIG_HEIGHT * FIG_SIZZE["mm"],
        aspect=FIG_ASPECT,
        legend=False,
        markers=True,
        style="ModelingMethod",
        ci=None,
        facet_kws={"sharey": True, "sharex": False},
        hue="ModelingMethod",
        palette=_pallet_and_order(hydra.glaph.compare_methods)[0],
        hue_order=_pallet_and_order(hydra.glaph.compare_methods)[1],
        # ).set(xlabel="Iteration", ylabel=f"Objective function",ylim=(-31*10**5,-28*10**5))
        # ).set(xlabel="Iteration", ylabel=f"Objective function",ylim=(-33*10**5,-30.4*10**5))
        # ).set(xlabel="Iteration", ylabel=f"Objective function",ylim=(-30.1*10**5,-28.4*10**5))
    ).set(
        xlabel="Iteration",
        ylabel=f"Objective function",
        # yscale="log",
        xlim=(ITR_MIN, ITR_MAX),
        ylim=(OBJ_MIN, OBJ_MAX),
        # title="",
        # lw=0.1,  # 線の太さ
        # markeredgewidth=0,
    )
    plt.subplots_adjust(hspace=0.4)  # 図同士の感覚を調整
    plt.legend(labels=_pallet_and_order(hydra.glaph.compare_methods)[1], loc="best")
    plt.savefig(
        os.path.join(res_root, f"cost_itr.pdf"),
    )


def cost_R_W(hydra, cost_df, res_root, id, start, end):
    cost_df = cost_df[(cost_df["Itration"] >= start)]
    cost_df = cost_df[(cost_df["Itration"] <= end)]
    ax = sns.relplot(
        data=cost_df,
        kind="line",
        x="Itration",
        y="tmp_cost",
        # y="CostW",
        row="Number of sources",
        height=FIG_HEIGHT * FIG_SIZZE["mm"],
        aspect=FIG_ASPECT,
        legend=False,
        markers=True,
        style="ModelingMethod",
        ci=None,
        facet_kws={"sharey": False, "sharex": False},
        hue="ModelingMethod",
        palette=_pallet_and_order(hydra.glaph.compare_methods)[0],
        hue_order=_pallet_and_order(hydra.glaph.compare_methods)[1],
        markevery=(0, 2),
        # ci='sd',
        # ).set(xlabel="Iteration", ylabel=f"Objective function", ylim=(-33*10**5,-30.4*10**5))
        # ).set(xlabel="Iteration", ylabel=f"Objective function", ylim=(-30.1*10**6,-0.5*10**6))
    ).set(
        xlabel="Iteration",
        ylabel=f"Objective function",
        # yscale="log",
        xlim=(ITR_MIN, ITR_MAX),
        ylim=(OBJ_MIN, OBJ_MAX),
    )
    # ).set(xlabel="Iteration", ylabel=f"Objective function")
    plt.legend(labels=_pallet_and_order(hydra.glaph.compare_methods)[1], loc="best")
    plt.savefig(os.path.join(res_root, f"cost_RW.pdf"))


def make_figure(hydra, res_root):
    df = make_df()
    cost_df = None
    sum_n_inc, sum_n_dec, sum_n = (
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
    )

    # データの読み込み
    # n_mic = "*"
    file_regex = "*"
    # file_regex = "[1-8]"
    # data_paths = glob.glob(f"{res_root}**/*res.csv", recursive=True)

    data_paths = glob.glob(
        f"{res_root}{n_mic}src_{n_mic}mic/ILRMA/{file_regex}/res.csv", recursive=True
    )
    data_paths += glob.glob(
        f"{res_root}{n_mic}src_{n_mic}mic/As_AE-Loss_*/{file_regex}/res.csv",
        recursive=True,
    )
    data_paths += glob.glob(
        f"{res_root}{n_mic}src_{n_mic}mic/ChimeraACVAE/{file_regex}/res.csv",
        recursive=True,
    )

    # try:
    for dp in data_paths:
        dry_df = pd.read_csv(dp)
        tmp_df = dry_df.groupby(
            by=[
                "Itration",
                "ModelingMethod",
                "NumberOfSources",
            ],
            as_index=False,
        ).mean()  # nchで平均

        tmp_df["CostI"] = tmp_df["CostI"] / tmp_df["NumberOfSources"]
        tmp_df["CostR"] = tmp_df["CostR"] / tmp_df["NumberOfSources"]
        tmp_df["CostW"] = tmp_df["CostW"] / tmp_df["NumberOfSources"]
        cost_init = tmp_df["CostI"].copy()
        cost_ipd_R = tmp_df["CostR"].copy()
        cost_upd_W = tmp_df["CostW"].copy()
        tmp_cost_df = return_concat_df(tmp_df)

        tmp_df["ImpCostUpdR"] = cost_ipd_R - cost_init
        tmp_df["ImpCostUpdW"] = cost_upd_W - cost_ipd_R

        df_imp_sdr = calc_imp(tmp_df)[0]
        df = df.append(df_imp_sdr)

    print("    Loading pandas data is done.")

    sns.set_style("whitegrid")
    id = "SI-SDR"
    df["NumberOfSources"] = df["NumberOfSources"].astype("int")

    # 手法名の整理
    df = df.replace("ChimeraACVAE", "FastMVAE")
    df = df.replace("As_AE-Loss_ISdiv", "Proposed_use_2ch")
    df = df.replace("As_AE-Loss_ISdiv_6ch", "Proposed_use_6ch")
    df = df.replace("As_AE-Loss_ISdiv_236ch", "Proposed_use_236ch")

    df = df.rename(columns={"NumberOfSources": "Number of sources"})
    df = df.groupby(
        by=[
            "ModelingMethod",
            "Number of sources",
            "CostW",
            "Itration",
            "ImpCostUpdR",
            "imp_sdr",
        ],
        as_index=False,
    ).mean()

    os.makedirs(res_root, exist_ok=True)
    fig = plt.figure()
    res_itr(hydra, df, res_root, id)
    fig = plt.figure()
    cost_itr(hydra, df, res_root, id, 1, 30)


def make_figure_each_file(hydra, res_root, res_each_file=None):
    df = make_df()
    cost_df = None
    sum_n_inc, sum_n_dec, sum_n = (
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
    )

    # データの読み込み
    if res_each_file == None:
        # data_paths = glob.glob(f"{res_root}**/*res.csv", recursive=True)
        data_paths = glob.glob(
            f"{res_root}{n_mic}src_{n_mic}mic/ILRMA/*/res.csv", recursive=True
        )
        data_paths += glob.glob(
            f"{res_root}{n_mic}src_{n_mic}mic/As_AE-Loss_ISdiv/*/res.csv",
            recursive=True,
        )
        data_paths += glob.glob(
            f"{res_root}{n_mic}src_{n_mic}mic/ChimeraACVAE/*/res.csv", recursive=True
        )

    else:
        for f in range(1, res_each_file):
            df = make_df()
            data_paths = glob.glob(
                f"{res_root}{n_mic}src_{n_mic}mic/ILRMA/{f}/res.csv", recursive=True
            )
            data_paths += glob.glob(
                f"{res_root}{n_mic}src_{n_mic}mic/As_AE-Loss_ISdiv*/{f}/res.csv",
                recursive=True,
            )
            data_paths += glob.glob(
                f"{res_root}{n_mic}src_{n_mic}mic/ChimeraACVAE/{f}/res.csv",
                recursive=True,
            )

            # try:
            for dp in data_paths:
                dry_df = pd.read_csv(dp)
                tmp_df = dry_df.groupby(
                    by=[
                        "Itration",
                        "ModelingMethod",
                        "NumberOfSources",
                    ],
                    as_index=False,
                ).mean()  # nchで平均

                tmp_n_inc, tmp_n_dec, tmp_n, method = _calc_rate_increase_obj(tmp_df)
                # sum_n_inc[METHODS[method]] += tmp_n_inc
                # sum_n_dec[METHODS[method]] += tmp_n_dec
                # sum_n[METHODS[method]] += tmp_n

                tmp_df["CostI"] = tmp_df["CostI"] / tmp_df["NumberOfSources"]
                tmp_df["CostR"] = tmp_df["CostR"] / tmp_df["NumberOfSources"]
                tmp_df["CostW"] = tmp_df["CostW"] / tmp_df["NumberOfSources"]
                cost_init = tmp_df["CostI"].copy()
                cost_ipd_R = tmp_df["CostR"].copy()
                cost_upd_W = tmp_df["CostW"].copy()
                tmp_cost_df = return_concat_df(tmp_df)

                tmp_df["ImpCostUpdR"] = cost_ipd_R - cost_init
                tmp_df["ImpCostUpdW"] = cost_upd_W - cost_ipd_R

                df_imp_sdr = calc_imp(tmp_df)[0]
                df = df.append(df_imp_sdr)
                # if cost_df is None:
                #    cost_df = tmp_cost_df
                # else:
                #    cost_df = cost_df.append(tmp_cost_df, ignore_index=True)

            print("    Loading pandas data is done.")

            sns.set_style("whitegrid")
            id = "SI-SDR"
            df["NumberOfSources"] = df["NumberOfSources"].astype("int")
            df = df.replace("As_AE-Loss_ISdiv", "Proposed_use_2ch")
            df = df.replace("As_AE-Loss_ISdiv_236ch", "Proposed_use_236ch")
            df = df.replace("ChimeraACVAE", "FastMVAE")

            df = df.rename(columns={"NumberOfSources": "Number of sources"})
            df = df.groupby(
                by=[
                    "ModelingMethod",
                    "Number of sources",
                    "CostW",
                    "Itration",
                    "ImpCostUpdR",
                    "imp_sdr",
                ],
                as_index=False,
            ).mean()

            # cost_df = cost_df.replace("As_AE-Loss_ISdiv", "Proposed")
            # cost_df = cost_df.rename(columns={"n_src": "Number of sources"})

            # fig = plt.figure()
            # res_itr_scat(hydra, df[["ModelingMethod", "Number of sources","Itration", "imp_sdr"]], res_root, id)

            os.makedirs(f"{res_root}/res_{f}", exist_ok=True)
            fig = plt.figure()
            res_itr(hydra, df, f"{res_root}/res_{f}", id)
            fig = plt.figure()
            cost_itr(hydra, df, f"{res_root}/res_{f}", id, 1, 30)


@hydra.main(config_name="hydras")
def main(hydra):
    hydra.arg.result_dir = hydra.arg.output_path
    make_figure(hydra, hydra.arg.result_dir)
    # make_figure_each_file(hydra, hydra.arg.result_dir, None)

    # 2ch 2 9 18 13
    # 3ch 8 17
