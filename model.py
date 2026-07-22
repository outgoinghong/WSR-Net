import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_wavelet_kernels(device=None, dtype=torch.float32):
    s = 1.0 / math.sqrt(2.0)
    h0 = torch.tensor([s, s], dtype=dtype, device=device)
    h1 = torch.tensor([-s, s], dtype=dtype, device=device)
    LL = torch.ger(h0, h0)
    LH = torch.ger(h0, h1)
    HL = torch.ger(h1, h0)
    HH = torch.ger(h1, h1)
    filt = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)
    return filt


class AdaptiveWaveletShrinkageAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.theta = nn.Parameter(torch.zeros(3, channels, 1, 1))

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        filt = build_wavelet_kernels()
        self.register_buffer("w_analysis", filt)
        self.register_buffer("w_synthesis", filt)

    def dwt(self, x):
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        weight = self.w_analysis.repeat(C, 1, 1, 1)
        y = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)
        y = y.view(B, C, 4, y.size(-2), y.size(-1)).contiguous()
        return y[:, :, 1], y[:, :, 2], y[:, :, 3], y[:, :, 0]

    def idwt(self, LH, HL, HH, LL):
        B, C, h, w = LL.shape
        y = torch.stack([LL, LH, HL, HH], dim=2).view(B, 4 * C, h, w)
        weight = self.w_synthesis.repeat(C, 1, 1, 1)
        return F.conv_transpose2d(y, weight=weight, bias=None, stride=2, padding=0, groups=C)

    @staticmethod
    def soft_threshold(x, thr):
        return torch.sign(x) * F.relu(torch.abs(x) - thr)

    def forward(self, x):
        B, C, H, W = x.shape
        LH, HL, HH, LL = self.dwt(x)

        eps = 1e-6
        m_LH = LH.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HL = HL.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HH = HH.abs().mean(dim=(2, 3), keepdim=True) + eps

        t = torch.sigmoid(self.theta)
        thr_LH = t[0].unsqueeze(0) * m_LH
        thr_HL = t[1].unsqueeze(0) * m_HL
        thr_HH = t[2].unsqueeze(0) * m_HH

        LH_hat = self.soft_threshold(LH, thr_LH)
        HL_hat = self.soft_threshold(HL, thr_HL)
        HH_hat = self.soft_threshold(HH, thr_HH)

        X_re = self.idwt(LH_hat, HL_hat, HH_hat, LL)
        if X_re.shape[-2:] != (H, W):
            X_re = X_re[..., :H, :W]

        attn_weight = self.attention(X_re)
        out = x * attn_weight + x
        return out

class SpectralChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        bottleneck_channels = max(1, in_channels // reduction_ratio)
        self.global_average_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.Sigmoid()

    def forward(self, input_feature):
        return input_feature * self.activation(self.channel_mlp(self.global_average_pool(input_feature)))


class SpatialRegionAttention(nn.Module):
    def __init__(self, attention_kernel_size=7):
        super().__init__()
        padding_size = attention_kernel_size // 2
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size=attention_kernel_size, padding=padding_size, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, input_feature):
        channel_average_map = torch.mean(input_feature, dim=1, keepdim=True)
        return input_feature * self.activation(self.spatial_conv(channel_average_map))


class CascadedSpectralSpatialJointPerception(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.spectral_attention_branch = SpectralChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention_branch = SpatialRegionAttention(spatial_kernel_size)
        self.spectral_spatial_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def _global_l2_normalize(self, correlation_matrix, eps=1e-6):
        l2_norm = torch.norm(correlation_matrix, p=2)
        return correlation_matrix / (l2_norm + eps)

    def forward(self, input_feature):
        batch_size, channels, height, width = input_feature.shape
        num_spatial_tokens = height * width

        spectral_aware_feature = input_feature + self.spectral_attention_branch(input_feature)
        spectral_tokens = spectral_aware_feature.view(batch_size, channels, num_spatial_tokens)

        spatial_aware_feature = input_feature + self.spatial_attention_branch(input_feature)
        spatial_tokens = spatial_aware_feature.view(batch_size, channels, num_spatial_tokens)

        spectral_to_spatial_correlation = torch.tanh(
            self._global_l2_normalize(torch.matmul(spectral_tokens.transpose(1, 2), spatial_tokens)))
        spatial_modulated_tokens = torch.matmul(spectral_tokens, spectral_to_spatial_correlation) + spatial_tokens

        spatial_to_spectral_correlation = torch.tanh(
            self._global_l2_normalize(torch.matmul(spatial_modulated_tokens.transpose(1, 2), spectral_tokens)))
        spectral_modulated_tokens = torch.matmul(spatial_modulated_tokens,
                                                 spatial_to_spectral_correlation) + spectral_tokens

        fused_feature = torch.cat([spectral_modulated_tokens.view(batch_size, channels, height, width),
                                   spatial_modulated_tokens.view(batch_size, channels, height, width)], dim=1)
        return self.spectral_spatial_fusion(fused_feature)


class PLMM_Decoder(nn.Module):
    def __init__(self, P, L):
        super().__init__()
        self.P = P
        self.L = L
        # 全局基准端元矩阵 (Base Endmembers)
        self.base_E = nn.Parameter(torch.rand(L, P))
        # 初始化为 (0, 1) 之间的均匀分布
        nn.init.uniform_(self.base_E, 0, 1)

    def forward(self, A, delta_E):

        B, P, H, W = A.shape


        base_E_expanded = self.base_E.view(1, self.L, self.P, 1, 1)


        E_dynamic = base_E_expanded + delta_E


        E_dynamic = F.relu(E_dynamic)

        A_expanded = A.unsqueeze(1)

        Y_hat = torch.sum(E_dynamic * A_expanded, dim=2)

        return Y_hat, E_dynamic


class WAANet(nn.Module):
    def __init__(self, P, L):
        super(WAANet, self).__init__()
        self.P = P
        self.L = L

        self.awsa = AdaptiveWaveletShrinkageAttention(channels=L)
        self.awsaA = AdaptiveWaveletShrinkageAttention(channels= P )
        self.feature_extractor = nn.Sequential(
            CascadedSpectralSpatialJointPerception(in_channels=L, reduction_ratio=16),
            nn.Conv2d(L, 120, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(120),
            CascadedSpectralSpatialJointPerception(in_channels=120, reduction_ratio=16),
            nn.Conv2d(120, 60, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(60)
        )

        self.abundance_branch = nn.Sequential(
            nn.Conv2d(60, self.P, kernel_size=3, padding=1),
            nn.Softmax(dim=1)  # 物理约束: ASC 和 ANC
        )
        self.perturbation_branch = nn.Sequential(
            nn.Conv2d(60, 128, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, self.L * self.P, kernel_size=1),
            nn.Tanh()  # 将扰动限制在 [-1, 1] 范围内
        )

        self.decoder = PLMM_Decoder(P=P, L=L)

    def forward(self, x):
        x_enhanced = self.awsa(x)

        deep_features = self.feature_extractor(x_enhanced)

        abu_est = self.abundance_branch(deep_features)
        B, _, H, W = deep_features.shape
        delta_E_flat = self.perturbation_branch(deep_features)

        delta_E = delta_E_flat.view(B, self.L, self.P, H, W)

        delta_E = delta_E * 0.005

        re_result, E_dynamic = self.decoder(abu_est, delta_E)

        return abu_est, re_result, delta_E

    def extract_base_endmembers(self):
        return self.decoder.base_E.data.squeeze()


if __name__ == "__main__":
    print("Initializing WAA-Net (PLMM Edition)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L, P = 156, 3
    H, W = 95, 95
    batch_size = 1

    model = WAANet(P=P, L=L).to(device)
    dummy_input = torch.randn(batch_size, L, H, W).to(device)

    abu_est, re_result, delta_E = model(dummy_input)

    print("\n[✔] Forward Pass Successful!")
    print(f"Input HSI Shape:          {dummy_input.shape}")
    print(f"Abundance Map (A):        {abu_est.shape}")
    print(f"Perturbation Map (ΔE):    {delta_E.shape}")
    print(f"Reconstructed Shape (Y^): {re_result.shape}")

    base_E = model.extract_base_endmembers()
    print(f"Base Endmembers (E):      {base_E.shape}")
