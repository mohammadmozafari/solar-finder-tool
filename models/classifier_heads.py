import torch
import numpy as np
import einops as eo
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
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
    
    def get_size_estimate(self, x_feats, vis=False, images=None, preds=None, start_num=None):
        patch_size = 14
        cls_embs, patch_embs = x_feats
        in_ = torch.cat([cls_embs[:, None, :], patch_embs], dim=1)
        _, attention = self.transformer.self_attn(in_, in_, in_, need_weights=True, average_attn_weights=False)
        attention = attention[:, :, 0, 1:]
        attention = eo.rearrange(attention, 'b h (ph pw) -> b h ph pw', ph=16, pw=16)
        attention = nn.functional.interpolate(attention, scale_factor=patch_size, mode = "nearest")
        attention = attention.sum(dim=1)            # (b, h, w)
        max_ = attention.amax((1, 2))
        min_ = attention.amin((1, 2))
        attention = (attention - min_[:, None, None]) / max_[:, None, None]
        attention = (attention > 0.4) * 1.0
        attention = attention.detach().cpu().numpy()

        h, w = attention.shape[-2:]

        if vis:
            
            for i in range(preds.shape[0]):
                
                if preds[i] < 0.5: continue
                                
                img = images[i].detach().cpu().numpy().transpose((1, 2, 0))
                normalized_img = Normalize(vmin=img.min(), vmax=img.max())(img)
                
                reds = plt.cm.Reds(attention[i])
                alpha_max_value = 1.00
                gamma = 0.5  
                
                rgba_img = np.zeros((h, w, 4))
                rgba_img[..., :3] = normalized_img
                rgba_img[..., 3] = 1
                
                rgba_mask = np.zeros((h, w, 4))
                rgba_mask[..., :3] = reds[..., :3]
                rgba_mask[..., 3] = np.power(attention[i], gamma) * alpha_max_value
                
                rgba_all = np.zeros((h, 2*w, 4))
                rgba_all[:, :w, :] = rgba_img
                rgba_all[:, w:, :] = rgba_mask
                
                rgba_all = Image.fromarray((rgba_all * 255).astype(np.uint8))
                rgba_all.save(f'tmp/attention_maps/{start_num + i}.png')
                

        counts = attention.sum((1, 2))
        return counts