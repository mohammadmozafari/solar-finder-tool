import torch
import torch.nn as nn
from matplotlib.colors import Normalize

class TransformerEncoderLinearHead(nn.Module):

    def __init__(self, d_model, output_size) -> None:
        super().__init__()
        # bug batch_first = True
        self.transformer = nn.TransformerEncoderLayer(d_model, 8, batch_first=True)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x_feats):
        cls_embs, patch_embs = x_feats
        in_ = torch.cat([cls_embs[:, None, :], patch_embs], dim=1)
        out_ = self.transformer(in_)
        out_ = self.fc(out_[:, 0, :])           # use different strategies
        return out_
    
    def get_size_estimate(self, x_feats):
        patch_size = 14
        cls_embs, patch_embs = x_feats
        in_ = torch.cat([cls_embs[:, None, :], patch_embs], dim=1)
        _, attention = self.transformer.self_attn(in_, in_, in_, need_weights=True, average_attn_weights=False)
        number_of_heads = attention.shape[1]
        attention = attention[0, :, 0, 1:].reshape(attention, -1)
        attention = attention.reshape(number_of_heads, 16, 16)
        attention = nn.functional.interpolate(attention.unsqueeze(0), scale_factor=patch_size, mode = "nearest")[0].cpu()
        attention_metric = attention.numpy()
        normalized_attention1_metric = Normalize(vmin=attention_metric.min(), vmax=attention_metric.max())(attention_metric)
        normalized_attention1_metric = (normalized_attention1_metric > 0.4) * 1.0
        count = normalized_attention1_metric.sum()
        return count