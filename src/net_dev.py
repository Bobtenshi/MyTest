import torch
import torch.nn as nn
from torch.autograd import Variable
from constant import EPSI, USE_NORM
import sys
# torch.full_like(input, fill_value, *, dtype=None
EPSItorch = torch.tensor(EPSI)


class SiLUConv2D(nn.Module):
    def __init__(
        self,
        input_ch,
        output_ch,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        use_norm=True,
        use_dropout=True,
    ):
        super(SiLUConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_ch, output_ch, kernel_size, stride, padding, dilation, groups, bias
        )
        self.silu = nn.SiLU()
        self.use_norm = use_norm
        self.use_dropout = use_dropout

        if self.use_norm:
            self.norm = nn.GroupNorm(1, output_ch)  # Layer Normalization
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.3)  # Layer Dropout
        nn.init.normal_(self.conv.weight, std=0.01)
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        h = self.conv(x)
        if self.use_norm:
            h = self.norm(h)
        if self.use_dropout:
            h = self.dropout(h)
        h = self.silu(h)

        return h


class SiLUDeconv2D(nn.Module):
    def __init__(
        self,
        input_ch,
        output_ch,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        use_norm=True,
    ):
        super(SiLUDeconv2D, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            input_ch,
            output_ch,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )
        self.silu = nn.SiLU()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.GroupNorm(1, output_ch)  # Layer Normalization

        nn.init.normal_(self.deconv.weight, std=0.01)
        if bias:
            nn.init.zeros_(self.deconv.bias)

    def forward(self, x):
        h = self.deconv(x)
        if self.use_norm:
            h = self.norm(h)
        h = self.silu(h)

        return h


# ========== VAE ==========
class CentaurVAE_Encoder(nn.Module):
    def __init__(self, n_freq, n_label, type="1D"):
        super(CentaurVAE_Encoder, self).__init__()
        self.type = type
        # 共通部分
        self.conv1 = SiLUConv2D(n_freq, n_freq // 2, (1, 5), (1, 1), (0, 2))
        self.conv2 = SiLUConv2D(n_freq // 2, n_freq // 4, (1, 4), (1, 2), (0, 1))

        # for mu, sigma
        self.conv3z = nn.Conv2d(n_freq // 4, n_freq // 8 * 2, (1, 4), (1, 2), (0, 1))

        # for class-label
        self.conv3c = SiLUConv2D(n_freq // 4, n_freq // 4, (1, 4), (1, 2), (0, 1))
        self.conv4c = nn.Conv2d(n_freq // 4, n_label, (1, 4), (1, 2), (0, 1))
        self.softmax = nn.Softmax(dim=1)

        nn.init.normal_(self.conv3z.weight, std=0.01)
        nn.init.zeros_(self.conv3z.bias)
        nn.init.normal_(self.conv4c.weight, std=0.01)
        nn.init.zeros_(self.conv4c.bias)

    def forward(self, x):
        if self.type == "1D":
            x = x.permute(0, 2, 1, 3)
        # 共通部分
        h = self.conv1(x)
        yz = self.conv2(h)

        # for mu, sigma
        z = self.conv3z(yz)
        mu, logvar = z.split(z.size(1) // 2, dim=1)

        # for class-label
        y = self.conv3c(yz)
        y = self.conv4c(y)
        y = self.softmax(y)
        y = torch.mean(y, dim=3)
        y = torch.squeeze(torch.clamp(y, 1.0e-35), dim=2)

        return mu, torch.clamp(logvar, max=0.0), y


class CentaurVAE_Decoder(nn.Module):
    def __init__(self, n_freq, n_label, type="1D"):
        super(CentaurVAE_Decoder, self).__init__()
        self.type = type
        self.deconv1 = SiLUDeconv2D(
            n_freq // 8 + n_label, n_freq // 4, (1, 4), (1, 2), (0, 1)
        )
        self.deconv2 = SiLUDeconv2D(
            n_freq // 4 + n_label, n_freq // 2, (1, 4), (1, 2), (0, 1)
        )
        self.deconv3 = nn.ConvTranspose2d(
            n_freq // 2 + n_label, n_freq, (1, 5), (1, 1), (0, 2)
        )

        nn.init.normal_(self.deconv3.weight, std=0.01)
        nn.init.zeros_(self.deconv3.bias)

    def concat_xy(self, x, y):
        n_h, n_w = x.shape[2:4]
        return torch.cat((x, y.unsqueeze(2).unsqueeze(3).repeat(1, 1, n_h, n_w)), dim=1)

    def forward(self, z, c):
        h = self.deconv1(self.concat_xy(z, c))
        h = self.deconv2(self.concat_xy(h, c))
        h = self.deconv3(self.concat_xy(h, c))

        if self.type == "1D":
            h = h.permute(0, 2, 1, 3)

        return torch.clamp(h, min=-80.0, max=15.0)


# ============= CentaurVAE traning ==============
# Proposed method: retrain only "encoder".
# ===============================================
class CentaurVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(CentaurVAE, self).__init__()
        self.encoder = encoder  # retrain
        self.decoder = decoder  # fix as ChimeraACVAE
        self.silu = nn.SiLU()

    def sample_z(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(), device=mu.device), requires_grad=True)
        # Reparameterization trick
        return mu + torch.exp(logvar / 2) * eps

    def add_gaussian_noize(self, x_abs):
        mu = torch.mean(x_abs, axis=(1, 2), keepdims=True)
        var1 = torch.mean((x_abs - mu) ** 2, axis=(1, 2), keepdims=True)
        var2 = torch.var(x_abs, axis=(1, 2), keepdims=True)
        print(f"{mu}, {var1[:3,0,0]}, {var2[:3,0,0]}")

    def forward(self, hydra, x_abs, s_abs, r, gv):
        self.P = torch.maximum(
            x_abs**2, EPSItorch
        )  # Pを計算 |wx|^2 (batch, 1, freq, time-frame)
        self.P_s = torch.maximum(
            s_abs**2, EPSItorch
        )  # P_sを計算 |wx|^2 (batch, 1, freq, time-frame)

        self.x_abs_norm = x_abs / torch.sqrt(gv)  # gvで正規化
        # spectrogram(x_abs_norm[0,0,:,:].to('cpu').detach().numpy().copy(), output_path="./in_spec")

        if hydra.params.retrain_model_type == "AE":
            self.z_mu, _, self.x_prob = self.encoder(self.x_abs_norm)
            # print(self.x_abs_norm[0, 0,0, 0])
            # print(self.z_mu[0, 0,0, 0])
            self.logvar = self.decoder(self.z_mu, self.x_prob)

        elif hydra.params.retrain_model_type == "VAE":
            self.z_mu, self.z_logvar, self.x_prob = self.encoder(self.x_abs_norm)
            z = self.sample_z(self.z_mu, self.z_logvar)
            self.logvar = self.decoder(z, self.x_prob)

        self.Q = torch.exp(self.logvar)  # 分散行列σ^2 (batch, 1, freq, time-frame)
        self.Q = torch.maximum(self.Q, EPSItorch)
        self.gv = torch.mean(
            torch.divide(self.P[:, :, :, :], self.Q[:, :, :, :]),
            axis=(1, 2, 3),
            keepdims=True,
        )  # (n_src, 1 ,1)
        self.Rhat = torch.multiply(self.Q, self.gv)  # 更新後の音源モデル
        self.R = r  # 更新前の音源モデル
        return None

    def label_snapshot(self):
        return self.x_prob

    def loss(self, hydra):
        # ============ calc IS-div (wx ll source-model) ==============

        #if hydra.params.retrain_model_type == "AE":
        if 'ISdiv' in  hydra.params.retrain_loss:
            losses = (self.P / self.Rhat) - torch.log(self.P) + torch.log(self.Rhat) - 1

            loss_mean = torch.mean(losses, [3, 2, 1, 0])  # [batch, 1, 1024, 128]
            loss = torch.sum(losses)
            return loss_mean, loss

        elif 'PIT' in  hydra.params.retrain_loss:
            #print(f"Rhat shape: {self.Rhat.shape}")
            #print(f"s_abs shape: {self.P_s.shape}")
            self.P_s_inv = torch.zeros_like(self.P_s)
            self.P_s_inv[0,:,:,:] = self.P_s[1,:,:,:]
            self.P_s_inv[1,:,:,:] = self.P_s[0,:,:,:]

            losses = (self.P_s / self.Rhat) - torch.log(self.P_s) + torch.log(self.Rhat) - 1
            losses_inv = (self.P_s_inv / self.Rhat) - torch.log(self.P_s_inv) + torch.log(self.Rhat) - 1

            #print(torch.sum(losses))
            #print(torch.sum(losses_inv))

            if torch.sum(losses) < torch.sum(losses_inv):
                return torch.mean(losses) , torch.sum(losses)
            else:
                return torch.mean(losses_inv) , torch.sum(losses_inv)

