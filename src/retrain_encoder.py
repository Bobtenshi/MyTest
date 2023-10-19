from constant import SEG_LEN, N_FREQ, MAX_BATCH_SIZE
from torch.utils.tensorboard import SummaryWriter
import net_dev
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import glob
from tqdm import tqdm
import hydra
from Tools.utils import get_root
from Tools.calc_sdr_by_tmp_dnn import calc_sdr_by_tmp_dnn_fake
from torch.utils.tensorboard import SummaryWriter

# import librosa
import sys
import random
import pandas as pd

random.seed(1)
df = pd.DataFrame(
    {
        "Epoch": [],
        "Train_loss": [],
        "Train2_loss": [],
        "Val_loss": [],
        "Test_loss": [],
        "Imp_SDR": [],
    }
)

batch_times = 1000
train_n = 10
eval_n = 10
test_n = 10


@hydra.main(config_name="hydras")
def _main(hydra):
    _, PROJECT_ROOT = get_root()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    class Params:
        pass

    config = Params()
    config.train_data = "./data/train/wsj0"
    config.test_data = "./data/test/jvs"
    config.teacher_chimera_model = "./model/trained/jvs/ChimeraACVAE/Chimera.model"
    config.epoch = 100
    config.snapshot = 1
    config.learning_rate = 0.00001
    config.pretrained_model = None
    config.batch_size = 1
    # log_dirでlogのディレクトリを指定
    writer = SummaryWriter(log_dir="./logs")
    n_epoch = config.epoch

    # Set input directories and data paths
    save_path = f"./model/trained/wsj0/As_{hydra.params.retrain_model_type}-Loss_{hydra.params.retrain_loss}/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # 学習用音声データ(jvs)をtrain, val用に分割
    data_paths = glob.glob(
        f"{config.train_data}/[2]src_?mic/**/*_.npz", recursive=True
    )  # 7500paths
    data_paths += (
        glob.glob(
            f"{config.train_data}/[2]src_?mic/**/*_itr[0-5]_nch*_.npz",
            recursive=True,
        )
        * 5
    )  # 1500*n paths 初期イタレーションを重点的に学習したい

    random.shuffle(data_paths)  # シャッフル
    data_border = int(len(data_paths) * 0.8)  # 8:2 -> Train:Val
    train_data_paths = data_paths[:data_border]
    val_data_paths = data_paths[data_border:]

    # テスト用音声データ(jvs)を取得
    test_data_paths_2ch = glob.glob(
        f"{config.test_data}/[2]src_?mic/[0-9]/**/*_.npz", recursive=True
    )
    # print(f"{config.test_data}/2src_2mic/")
    # print(test_data_paths_2ch)
    ##test_data_paths_3ch = glob.glob("./output/trained/wsj0/more_spk_ind/3src_3mic/ChimeraACVAE/**/*_.npz", recursive=True)

    # Set up model and optimizer
    n_src = 101
    nn_freq = N_FREQ - 1
    # load trained-chimara-ACvae networks
    encoder = net_dev.CentaurVAE_Encoder(nn_freq, n_src)
    decoder = net_dev.CentaurVAE_Decoder(nn_freq, n_src)
    checkpoint = torch.load(config.teacher_chimera_model, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    # model to cuda
    for para in decoder.parameters():
        para.requires_grad = False

    # docoderの固定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder.eval()
    decoder.eval()

    # define CentaurVAE model
    model = net_dev.CentaurVAE(encoder, decoder)

    # GPUが使えるなら使用
    if device == "cuda":
        encoder.cuda(device)
        decoder.cuda(device)
        model.cuda(device)

    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=config.learning_rate)

    if config.pretrained_model == "" or config.pretrained_model is None:
        start_epoch = 0
    else:
        start_epoch = 0
        writer.add_text("log", "The pretrained model does not exist.")

    # set cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # define functions
    def model_snapshot(epoch):
        print(f"save the model at {epoch} epoch")
        torch.save(
            {
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "encoder_optimizer_state_dict": optimizer_enc.state_dict(),
                "decoder_optimizer_state_dict": optimizer_dec.state_dict(),
            },
            os.path.join(save_path, f"{epoch}.model"),
        )

    def loss_snapshot(hydra, train_losslist, val_losslist, test_2ch_losslist, sdr_list):
        colorlist = ["#4d5057", "#4d5057", "#35589A", "#370665"]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        length = len(train_losslist)
        epo = np.linspace(0, length, length) * config.snapshot

        ax1.plot(epo, train_losslist, label="train", color=colorlist[0])
        ax1.plot(
            epo, val_losslist, label="valid", color=colorlist[1], linestyle="dashed"
        )
        ax1.plot(epo, test_2ch_losslist, label="test_2ch_data", color=colorlist[2])
        ax1.legend(
            bbox_to_anchor=(1, 1), loc="upper right", borderaxespad=1, fontsize=18
        )

        ax2 = ax1.twinx()
        ax2.plot(epo, sdr_list, label="imp_SI-SDR", color=colorlist[3], marker="o")
        ax2.legend(
            bbox_to_anchor=(1, 1), loc="lower right", borderaxespad=1, fontsize=18
        )

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)
        ax2.set_ylabel("imp SI-SDR")

        fig.set_size_inches(10, 8)
        fig.savefig(os.path.join(save_path, f"loss_sdr.pdf"))
        plt.close()

    train_losslist = []
    val_losslist = []
    test_2ch_losslist = []
    test_3ch_losslist = []
    sdr_list = []
    # train

    # calc sdr by chimera dnn-model
    sdr_chimera = calc_sdr_by_tmp_dnn_fake(hydra, config.teacher_chimera_model)
    # sdr_chimera = 0
    df_curv = df
    try:
        for epoch in tqdm(range(start_epoch, n_epoch + 1)):
            train_loss_in_epoch = []
            train2_loss_in_epoch = []
            val_loss_in_epoch = []
            test_2ch_loss_in_epoch = []

            #############################
            # Train-mode use JVS data-set
            #############################
            model.train()
            random.shuffle(train_data_paths)  # 7,000 fileぐらい
            train_data = train_data_paths[: config.batch_size * batch_times]
            for start in range(
                0, len(train_data), config.batch_size
            ):  # 16-24バッチぐらいでぶんかつ
                #print(start)
                batch_wide_data_path = train_data_paths[
                    start : start + config.batch_size
                ]  # batch_wide_data_path
                res = None

                for data_path in batch_wide_data_path:  # 各data_pathのデータを結合して1バッチとする
                    if res == None:
                        res = np.load(data_path)  # load
                        x_abs = res["Y_abs"]  # 分離途中の信号の絶対値 |y| (batch,freq,time-frame)
                        s_abs = res["S_abs"][
                            :, :, :128
                        ]  # 正解信号の絶対値 |y| (batch,freq,time-frame)
                        r = res["source_model"]  # 音源モデル  (batch,freq,time-frame)
                        gv = res["gv"]  # |y|の平均値 chごと (batch,1,1)

                        # print(x_abs.shape)
                        # print(s_abs.shape)
                        # print(np.mean(x_abs))
                        # print(np.mean(abs(s_abs)))

                    else:  # バッチサイズを満たすまで結合
                        res = np.load(data_path)  # load

                        x_abs = np.concatenate(
                            [x_abs, res["Y_abs"]], axis=0
                        )  # 分離途中の信号の絶対値 |y| (batch,freq,time-frame)
                        s_abs = np.concatenate(
                            [s_abs, res["S_abs"][:, :, :128]], axis=0
                        )  # 分離途中の信号の絶対値 |y| (batch,freq,time-frame)

                        r = np.concatenate(
                            [r, res["source_model"]], axis=0
                        )  # 音源モデル  (batch,freq,time-frame)
                        gv = np.concatenate(
                            [gv, res["gv"]], axis=0
                        )  # |y|の平均値 chごと (batch,1,1)

                # to torch on GPU
                x_abs = torch.from_numpy(
                    np.asarray(x_abs[:, None, 1:, :], dtype="float32")
                ).to(device)
                s_abs = torch.from_numpy(
                    np.asarray(s_abs[:, None, 1:, :], dtype="float32")
                ).to(device)

                r = torch.from_numpy(np.asarray(r[:, None, 1:, :], dtype="float32")).to(
                    device
                )
                gv = torch.from_numpy(
                    np.asarray(gv[:, None, :, :], dtype="float32")
                ).to(device)

                # update trainable parameters
                optimizer_enc.zero_grad()
                optimizer_dec.zero_grad()

                # Ono ver.
                model(hydra, x_abs, s_abs, r, gv)
                losses_mean, losses = model.loss(hydra)

                if epoch > 0:
                    losses.backward()
                    optimizer_enc.step()  # Enc.更新

                # optimizer_dec.step() # Dec. 更新しない
                train_loss_in_epoch.append(
                    losses_mean
                )  # [2.2700, 2.2604, 2.6442, 2.8901]
                model(hydra, x_abs, s_abs, r, gv)
                losses_mean2, losses2 = model.loss(hydra)

                del losses, losses_mean  # 誤差逆伝播を実行後、計算グラフを削除

            ##########################
            # train-mode use wsj0 data-set
            ##########################
            model.eval()  ## torch.nn.Module.eval
            with torch.no_grad():  ## disable autograd
                for data_path in train_data_paths[:train_n]:  # 2000fileぐらい
                    # for data_path in train_data[: config.batch_size * 1]:  # 2000fileぐらい
                    res = np.load(data_path)  # load
                    x_abs = res["Y_abs"]  # 分離途中の信号の絶対値
                    s_abs = res["S_abs"][
                        :, :, :128
                    ]  # 正解信号の絶対値 |y| (batch,freq,time-frame)
                    r = res["source_model"]  # 分離途中の信号の絶対値
                    gv = res["gv"]
                    # spectrogram((x_abs/np.sqrt(gv))[0,:,:], output_path="./val_in_spec")

                    x_abs = torch.from_numpy(
                        np.asarray(x_abs[:, None, 1:, :], dtype="float32")
                    ).to(device)
                    s_abs = torch.from_numpy(
                        np.asarray(s_abs[:, None, 1:, :], dtype="float32")
                    ).to(device)
                    r = torch.from_numpy(
                        np.asarray(r[:, None, 1:, :], dtype="float32")
                    ).to(device)
                    gv = torch.from_numpy(
                        np.asarray(gv[:, None, :, :], dtype="float32")
                    ).to(device)
                    # Ono ver.
                    model(hydra, x_abs, s_abs, r, gv)
                    losses_mean, losses = model.loss(hydra)
                    train2_loss_in_epoch.append(losses_mean)
                    del losses, losses_mean  # 誤差逆伝播を実行後、計算グラフを削除

            ##########################
            # Val-mode use wsj0 data-set
            ##########################
            model.eval()  ## torch.nn.Module.eval
            with torch.no_grad():  ## disable autograd
                for data_path in val_data_paths[:eval_n]:  # 2000fileぐらい
                    # for data_path in train_data[: config.batch_size * 1]:  # 2000fileぐらい
                    res = np.load(data_path)  # load
                    x_abs = res["Y_abs"]  # 分離途中の信号の絶対値
                    s_abs = res["S_abs"][
                        :, :, :128
                    ]  # 正解信号の絶対値 |y| (batch,freq,time-frame)
                    r = res["source_model"]  # 分離途中の信号の絶対値
                    gv = res["gv"]
                    # spectrogram((x_abs/np.sqrt(gv))[0,:,:], output_path="./val_in_spec")

                    x_abs = torch.from_numpy(
                        np.asarray(x_abs[:, None, 1:, :], dtype="float32")
                    ).to(device)
                    s_abs = torch.from_numpy(
                        np.asarray(s_abs[:, None, 1:, :], dtype="float32")
                    ).to(device)
                    r = torch.from_numpy(
                        np.asarray(r[:, None, 1:, :], dtype="float32")
                    ).to(device)
                    gv = torch.from_numpy(
                        np.asarray(gv[:, None, :, :], dtype="float32")
                    ).to(device)
                    # Ono ver.
                    model(hydra, x_abs, s_abs, r, gv)
                    losses_mean, losses = model.loss(hydra)
                    val_loss_in_epoch.append(losses_mean)
                    del losses, losses_mean  # 誤差逆伝播を実行後、計算グラフを削除

            ###############################
            # Test-mode use WSJ0 data-set #
            ###############################
            model.eval()  ## torch.nn.Module.eval
            with torch.no_grad():  ## disable autograd
                for data_path in test_data_paths_2ch[:test_n]:  # 2000fileぐらい
                    res = np.load(data_path)  # load
                    x_abs = res["Y_abs"]  # 分離途中の信号の絶対値
                    s_abs = res["S_abs"][
                        :, :, :128
                    ]  # 正解信号の絶対値 |y| (batch,freq,time-frame)
                    r = res["source_model"]  # 分離途中の信号の絶対値
                    gv = res["gv"]

                    x_abs = torch.from_numpy(
                        np.asarray(x_abs[:, None, 1:, :], dtype="float32")
                    ).to(device)
                    s_abs = torch.from_numpy(
                        np.asarray(s_abs[:, None, 1:, :], dtype="float32")
                    ).to(device)
                    r = torch.from_numpy(
                        np.asarray(r[:, None, 1:, :], dtype="float32")
                    ).to(device)
                    gv = torch.from_numpy(
                        np.asarray(gv[:, None, :, :], dtype="float32")
                    ).to(device)
                    # Ono ver.
                    model(hydra, x_abs, s_abs, r, gv)
                    losses_mean, losses = model.loss(hydra)
                    test_2ch_loss_in_epoch.append(losses_mean)
                    del losses, losses_mean  # 誤差逆伝播を実行後、計算グラフを削除

            # save model
            model_snapshot(epoch)

            if epoch % config.snapshot == 0:  # 10エポックでLossグラフ作成
                sdr_in_epoch = calc_sdr_by_tmp_dnn_fake(
                    hydra, os.path.join(save_path, f"{epoch}.model")
                )
                df_curv = df_curv.append(
                    {
                        "Epoch": int(epoch),
                        "Train_loss": torch.mean(
                            torch.Tensor(train_loss_in_epoch)
                        ).item(),
                        "Train2_loss": torch.mean(
                            torch.Tensor(train2_loss_in_epoch)
                        ).item(),
                        "Val_loss": torch.mean(torch.Tensor(val_loss_in_epoch)).item(),
                        "Test_loss": torch.mean(
                            torch.Tensor(test_2ch_loss_in_epoch)
                        ).item(),
                        "Imp_SDR": (sum(sdr_in_epoch) - sum(sdr_chimera))
                        / len(sdr_in_epoch),
                    },
                    ignore_index=True,
                )

                # epoch内のLossから平均をとって格納→グラフ用に
                # train_losslist.append(torch.mean(torch.Tensor(train_loss_in_epoch)))
                # val_losslist.append(torch.mean(torch.Tensor(val_loss_in_epoch)))
                # test_2ch_losslist.append(
                #    torch.mean(torch.Tensor(test_2ch_loss_in_epoch))
                # )
                #
                ## calc sdr by tmp dnn-model
                # sdr_in_epoch = calc_sdr_by_tmp_dnn_fake(
                #    hydra, os.path.join(save_path, f"{epoch}.model")
                # )
                # sdr_list.append(
                #    (sum(sdr_in_epoch) - sum(sdr_chimera)) / len(sdr_in_epoch)
                # )
                # loss_snapshot(
                #    hydra, train_losslist, val_losslist, test_2ch_losslist, sdr_list
                # )

                # tensorboad
                writer.add_scalar(
                    "Train loss", torch.mean(torch.Tensor(train_loss_in_epoch)), epoch
                )
                writer.add_scalar(
                    "Train2 loss", torch.mean(torch.Tensor(train2_loss_in_epoch)), epoch
                )
                writer.add_scalar(
                    "Valid loss", torch.mean(torch.Tensor(val_loss_in_epoch)), epoch
                )
                writer.add_scalar(
                    "Test loss", torch.mean(torch.Tensor(test_2ch_loss_in_epoch)), epoch
                )
                writer.add_scalar(
                    "SI-SDR imp.",
                    (sum(sdr_in_epoch) - sum(sdr_chimera)) / len(sdr_in_epoch),
                    epoch,
                )
            df_curv.to_csv(os.path.join(save_path, "train_curv.csv"), index=False)

    except KeyboardInterrupt:
        writer.add_text("error", "\nKeyboard interrupt, exit.")

    else:
        writer.add_text("success", "Training done!")

    finally:
        writer.add_text("model", f"Output: {save_path}")
        writer.close()


if __name__ == "__main__":
    _main()
