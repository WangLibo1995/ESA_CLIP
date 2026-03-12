import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import timm
from open_clip import create_model_and_transforms, tokenizer, CLIPTextCfg

CLASSES = ['blue algae', 'bushfire', 'debris flow', 'farmland fire','flood',
           'forest fire', 'green tide', 'red tide', 'volcanic eruption', 'normal']

class SpatialAdapter(nn.Module):
    def __init__(self, in_dim, num_branches=4, reduction=4):
        super().__init__()
        branch_channels = in_dim // num_branches
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, groups=in_dim, bias=False),
            nn.Conv2d(in_dim, branch_channels, 1)
        )
        self.atrous_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(branch_channels, branch_channels, 3, padding=d, dilation=d,
                          groups=branch_channels, bias=False),
                nn.Conv2d(branch_channels, branch_channels, 1),
                nn.ReLU(),
                nn.Dropout2d(p=0.2)
            ) for d in (1, 2, 3)
        ])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // reduction, in_dim, bias=True),
            nn.Dropout(p=0.3)
        )
        self.ln = nn.LayerNorm(in_dim)

    def forward(self, x):
        B, N, D = x.shape
        H = W = int((N-1)**0.5)
        cls_token = x[:, 0, :]  # [B, D]
        img_tokens = x[:, 1:, :]    # [B, HW, D]

        img_features = img_tokens.transpose(1, 2).contiguous().view(B, D, H, W)
        b1 = self.conv1x1(img_features)
        b2, b3, b4 = [atrous(b1) for atrous in self.atrous_blocks]
        ms_features = torch.cat([b1, b2, b3, b4], dim=1)
        img_tokens_upd = ms_features.view(B, -1, D)

        gap_feat = self.gap(ms_features).view(B, D)
        cls_token_upd = cls_token + self.fc(gap_feat)
        cls_token_upd = cls_token_upd.view(B, 1, D)

        x_upd = torch.cat([cls_token_upd, img_tokens_upd], dim=1)
        return self.ln(x_upd)


class SpectrumAdapter(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear_diff = nn.Linear(in_dim, in_dim, bias=True)
        self.ln = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4, bias=True),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_dim * 4, in_dim, bias=True)
        )
        self.ln = nn.LayerNorm(in_dim)

    def sid_matrix(self, q, k):
        q = F.tanh(q) * 0.499 + 0.5
        k = F.tanh(k) * 0.499 + 0.5
        eps = 1e-8
        q_prob = torch.clamp(q / (q.sum(dim=-1, keepdim=True) + eps), min=eps)
        k_prob = torch.clamp(k / (k.sum(dim=-1, keepdim=True) + eps), min=eps)
        log_ratio = torch.log(q_prob) - torch.log(k_prob)
        sid = torch.matmul(q_prob, log_ratio.transpose(-2, -1)) + torch.matmul(k_prob, -log_ratio.transpose(-2, -1))
        return -sid

    def forward(self, q, k, v):
        sid_sim = self.sid_matrix(q, k)
        attn_weights = F.softmax(sid_sim / q.size(-1)**0.5, dim=-1)
        attn_feat = torch.matmul(attn_weights, v)

        f_diff = self.linear_diff(v - attn_feat) + q
        out = self.mlp(self.ln(f_diff)) + f_diff
        return self.ln(out)

class DynamicWeightFusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_dim, in_dim *2, 1),
            nn.BatchNorm2d(in_dim *2),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_dim *2, in_dim, 3, padding=1),
            nn.Sigmoid()
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_norm = nn.LayerNorm(in_dim * 2)
        self.conv1d = nn.Conv1d(1, 1, 3, padding=1)
        self.fc1 = nn.Linear(in_dim * 2, in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.Dropout(p=0.4)  # 输出层前添加较高Dropout
        )
        self.ln = nn.LayerNorm(in_dim)

    def forward(self, x_rgb):
        B, N, D = x_rgb.shape
        H = W = int((N - 1) ** 0.5)
        rgb_cls = x_rgb[:, 0, :]
        rgb_tokens = x_rgb[:, 1:, :]
        rgb_feat = rgb_tokens.reshape(B, D, H, W)
        gate = self.gate(rgb_feat)
        img_fused = torch.mul(rgb_feat, gate)
        img_gap = self.gap(img_fused).view(B, D)
        img_feat = torch.cat([rgb_cls, img_gap], dim=1)   # [B, 3D]

        img_feat = self.channel_norm(img_feat)
        residual = img_feat
        global_feat= self.fc2(self.fc1(img_feat))
        local_feat = self.conv1d(img_feat.view(B, 1, 2*D)).view(B, 2*D)
        fused = F.softmax(global_feat, dim=-1) * img_feat + \
                F.softmax(local_feat, dim=-1) * img_feat + \
                residual
        return self.ln(self.fc(fused))

class Adapter(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.spatial_adapter1 = SpatialAdapter(in_dim)
        self.spatial_adapter2 = SpatialAdapter(in_dim)
        self.ln = nn.LayerNorm(in_dim)
        self.rgb_to_qkv = nn.Linear(in_dim, in_dim * 3)
        self.swir_to_qkv = nn.Linear(in_dim, in_dim * 3)
        self.spectrum_adapter1 = SpectrumAdapter(in_dim)
        self.spectrum_adapter2 = SpectrumAdapter(in_dim)
        self.fusionizer = DynamicWeightFusion(in_dim)

    def forward(self, x_rgb):
        rgb_spatial = self.spatial_adapter1(x_rgb)
        # swir_spatial = self.spatial_adapter2(x_swir)

        rgb_spatial = self.ln(rgb_spatial)
        # swir_spatial = self.ln(swir_spatial)
        q_rgb, k_rgb, v_rgb = self.rgb_to_qkv(rgb_spatial).chunk(3, dim=-1)
        # q_swir, k_swir, v_swir = self.swir_to_qkv(swir_spatial).chunk(3, dim=-1)
        rgb_spectrum = self.spectrum_adapter1(q_rgb, k_rgb, v_rgb)
        # swir_spectrum = self.spectrum_adapter2(q_rgb, k_swir, v_swir)

        x = self.fusionizer(rgb_spectrum)
        return x



class CustomTextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.classnames = CLASSES
        self.clip_model = clip_model
        self.temp1 = 'This is a multispectral satellite image depicting {}, a type of earth surface anomaly, captured at 10-meter spatial resolution. It contains six spectral bands (blue, green, red, near-infrared, shortwave infrared 1, and shortwave infrared 2) approximately covering 450 to 2500 nanometers range.'
        self.temp2 = 'This is a normal multispectral satellite image. No earth surface anomalies were detected.'

    def forward(self, img):
        prompts = []
        for c in self.classnames:
            if c == 'normal':
                prompts.append(self.temp2)
            else:
                prompts.append(self.temp1.format(c))
        text = tokenizer.tokenize(prompts).to(img.device)
        text_features = self.clip_model.encode_text(text, normalize=True)

        return text_features


class ESA_CLIP(nn.Module):
    def __init__(
        self,
        model_name='ViT-B-32',
        pretrained_name='laion2b_s34b_b79k',
        use_indice=False,
        use_adapter= False
    ):
        super().__init__()
        self.use_indice = use_indice
        self.use_adapter = use_adapter
        self.clip, _, _ = create_model_and_transforms(model_name, pretrained=pretrained_name, force_image_size=1024)
        if self.use_adapter == True:
            self.clip.visual.output_tokens = True   # 默认False. 设置后，visual(img) 返回 (pooled, tokens)，tokens 包含 CLS + 所有 patch
            self.clip.visual.pool_type = 'none' # 默认为 pool_type='tok'.包含 CLS 与 patch tokens :contentReference[oaicite:2]{index=2}
            self.dim_hidden = CLIPTextCfg.width # self.clip.visual.transformer.width
            self.adapter = Adapter(self.dim_hidden)
            self.ln = nn.LayerNorm(self.dim_hidden)

        else:
            self.fc_adapter = nn.Linear(512, 512, bias=False)

        self.freeze()
        self.text_encoder = CustomTextEncoder(self.clip)


    def freeze(self):
        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

    def pretrain_norm(self, img):
        normalize = torchvision.transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])
        return normalize(img)

    def forward(self, img_rgb):
        # img_rgb = self.pretrain_norm(img_rgb)
        # img_swir = self.pretrain_norm(img_swir)

        if self.use_adapter:
            rgb_tokens, _ = self.clip.visual(img_rgb)   # 前512，后768
            # swir_tokens, _ = self.clip.visual(img_swir)
            rgb_tokens = self.ln(rgb_tokens)
            # swir_tokens = self.ln(swir_tokens)
            image_features = self.adapter(rgb_tokens)

        else:
            img_rgb_feat = self.clip.encode_image(img_rgb)  # (batch_size, 512)
            image_features = self.fc_adapter(img_rgb_feat)

        # 对比学习
        image_features = F.normalize(image_features, dim=-1) # L2归一化
        text_features = F.normalize(self.text_encoder(img_rgb))
        logits = self.clip.logit_scale.exp() * image_features @ text_features.T

        res = {"logits": logits} #{"image_features": image_features, "text_features": text_features, "logits": logits}
        return res






