import numpy as np
import torch
import torch.nn as nn

# the code of 3D position encoding is referred from https://github.com/tatp22/multidim-positional-encoding

def get_emb(sin_inp):
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.device = "cuda"

    def forward(self, batch, xyz):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        self.cached_penc = torch.zeros((batch, self.org_channels), device=self.device, dtype=self.inv_freq.dtype,)
        sin_inp_x = torch.einsum("i,j->ij", x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", z, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb_y = get_emb(sin_inp_y)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (batch, self.channels * 3),
            device=self.device,
            dtype=self.inv_freq.dtype,
        )
        emb[:, : self.channels] = emb_x
        emb[:, self.channels : 2 * self.channels] = emb_y
        emb[:, 2 * self.channels :] = emb_z
        self.cached_penc = emb[:, :self.org_channels]

        return self.cached_penc


class Summer(nn.Module):
    def __init__(self, penc):
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor, batch, xyz):
        penc = self.penc(batch, xyz)[:, :, None]
        penc = penc.to(tensor.device)
        return tensor + penc


def learnable_embedding(pos_len, dim):
    pos_emb = nn.Embedding(pos_len, dim)
    return pos_emb
