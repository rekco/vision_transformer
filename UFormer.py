from torch import nn, Tensor
from ViT import LocalFormer


class Encoder(nn.Module):
    def __init__(self,
                 in_chs: int = 3,
                 im_size: int = 224,
                 depth: int = 2,
                 large_patch_size: int = 16,
                 large_emb_size: int = 1024,
                 large_num_heads: int = 8,
                 large_att_drop_p: float = 0.,
                 large_ffn_expansion: int = 4,
                 large_ffn_drop_p: float = 0.,
                 small_patch_size: int = 8,
                 small_emb_size: int = 1024,
                 small_num_heads: int = 8,
                 small_att_drop_p: float = 0.,
                 small_ffn_expansion: int = 4,
                 small_ffn_drop_p: float = 0.
                 ):
        super(Encoder, self).__init__()
        self.large_former = LocalFormer(in_chs,
                                        im_size,
                                        depth,
                                        large_patch_size,
                                        large_emb_size,
                                        large_num_heads,
                                        large_att_drop_p,
                                        large_ffn_expansion,
                                        large_ffn_drop_p)
        self.small_former = LocalFormer(in_chs,
                                        im_size,
                                        depth,
                                        small_patch_size,
                                        small_emb_size,
                                        small_num_heads,
                                        small_att_drop_p,
                                        small_ffn_expansion,
                                        small_ffn_drop_p)

    def forward(self, x: Tensor) -> Tensor:
        x = self.large_former(x)
        x = self.small_former(x)

        return x


class DownSample(nn.Sequential):
    def __init__(self, in_chs, out_chs):
        super(DownSample, self).__init__(
            nn.Upsample(scale_factor=0.5),
            nn.Conv2d(in_chs, out_chs, kernel_size=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU())


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        pass



