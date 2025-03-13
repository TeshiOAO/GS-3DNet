import torch
import torch.nn as nn
from utils import position_encodings

class FCModel(nn.Module):
    def __init__(self, batch, hid_c, out_c):
        super().__init__()
        
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.backends.cudnn.benchmark = True
        else:
            self.device = "cpu"

        self.batch = batch
        self.in_channels = hid_c
        self.hid_c = hid_c
        self.out_c = out_c
        self.blocksize = 8

        self.p_enc_model = position_encodings.PositionalEncoding3D(hid_c).to(self.device)
        self.positions = nn.Embedding(hid_c, hid_c//3)

        self.p_emb = nn.Sequential(
            nn.Conv1d(1, hid_c//3, 1),
            nn.SiLU(1),
        )

        self.pos_encode = nn.Sequential(
            nn.Conv3d(hid_c, hid_c, (1,1,1)),
            nn.SiLU(1),
        )

        self.g0 =  nn.Sequential(
            nn.Conv3d(1, hid_c, (1,1,1)),
            nn.SiLU(1),
        )
        self.g1 = nn.Sequential(
            nn.Conv3d(hid_c, hid_c, (1,1,1)),
            nn.SiLU(1),
            nn.LayerNorm([self.blocksize, self.blocksize, self.blocksize]),
        )
        self.g2 = nn.Sequential(
            nn.Conv3d(hid_c, hid_c//2, (2,2,2)),
            nn.SiLU(1),
            nn.LayerNorm([self.blocksize-1, self.blocksize-1, self.blocksize-1]),
        )

        self.g_1 = nn.Sequential(
            nn.ConvTranspose3d(hid_c//2, hid_c, (2,2,2)),
            nn.SiLU(1),
        )
        self.g_2 = nn.Sequential(
            nn.Conv3d(hid_c, out_c, (1,1,1)),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, xyz, param):
        aggr_pos = self.p_enc_model(self.batch, xyz).view(self.batch, self.hid_c, 1,1,1)
        aggr_pos_enc = self.pos_encode(aggr_pos)
        aggr_pos_enc -= aggr_pos_enc.min(1, keepdim=True)[0]
        aggr_pos_enc /= aggr_pos_enc.max(1, keepdim=True)[0]
        aggr_pos_enc = torch.nan_to_num(aggr_pos_enc, nan=0)

        p = self.p_emb(param[:, None, :]).view(self.batch, self.hid_c)
        p -= p.min(1, keepdim=True)[0]
        p /= p.max(1, keepdim=True)[0]
        p = p.view(self.batch, self.hid_c, 1, 1, 1)
        p = torch.nan_to_num(p, nan=0)

        h = x.view(self.batch, 1, self.blocksize, self.blocksize, self.blocksize)
        h = self.g0(h)

        ### parameter/timestep embedding
        h += p
        h = self.g1(h)

        ### postion embedding
        h += aggr_pos_enc
        h = self.g1(h)

        h = self.g2(h)

        h = self.g_1(h)
        h = self.g_2(h)

        h = h.view(self.batch, self.out_c, self.blocksize**3)
        log_alpha = h.transpose(1,2).contiguous()

        return log_alpha