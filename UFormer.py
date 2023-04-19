from torch import nn, Tensor
from model.ViT import Transformer


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_chs: int = 3,
                 im_size: int = 224,
                 num_blocks: int = 2,
                 block_depth: int = 2,
                 large_patch_size: int = 14,
                 large_emb_size: int = 1024,
                 large_num_heads: int = 8,
                 large_att_drop_p: float = 0.,
                 large_ffn_expansion: int = 4,
                 large_ffn_drop_p: float = 0.,
                 small_patch_size: int = 7,
                 small_emb_size: int = 1024,
                 small_num_heads: int = 8,
                 small_att_drop_p: float = 0.,
                 small_ffn_expansion: int = 4,
                 small_ffn_drop_p: float = 0.
                 ):
        super(EncoderBlock, self).__init__()
        transformers = []
        for i in range(num_blocks):
            transformers += [
                Transformer(in_chs, im_size, block_depth, large_patch_size, large_emb_size, large_num_heads,
                            large_att_drop_p, large_ffn_expansion, large_ffn_drop_p),
                Transformer(in_chs, im_size, block_depth, small_patch_size, small_emb_size, small_num_heads,
                            small_att_drop_p, small_ffn_expansion, small_ffn_drop_p)
            ]
        self.transformers = nn.Sequential(*transformers)

    def forward(self, x: Tensor) -> Tensor:
        for transformer in self.transformers:
            x = transformer(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 skip_chs: int,
                 up_chs: int,
                 out_chs: int):
        super(DecoderBlock, self).__init__()

    def forward(self, x):
        pass


class DownSample(nn.Sequential):
    def __init__(self, in_chs, out_chs):
        super(DownSample, self).__init__(
            nn.Upsample(scale_factor=0.5),
            nn.Conv2d(in_chs, out_chs, kernel_size=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU()
        )


class UpSample(nn.Sequential):
    def __init__(self, in_chs, out_chs):
        super(UpSample, self).__init__(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_chs, out_chs, kernel_size=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU()
        )


class UFormer(nn.Module):
    def __init__(self,
                 in_chs: int = 3,
                 out_chs: int = 1,
                 im_size: int = 224):
        super(UFormer, self).__init__()
        self.encoder1 = EncoderBlock(in_chs=in_chs, im_size=im_size, num_blocks=2)

        self.down_sample1 = DownSample(3, 16)
        self.encoder2 = EncoderBlock(in_chs=16, im_size=int(im_size/2), num_blocks=2)

        self.down_sample2 = DownSample(16, 32)
        self.encoder3 = EncoderBlock(in_chs=32, im_size=int(im_size/4), num_blocks=6)

        self.down_sample3 = DownSample(32,  64)
        self.bottleneck = Transformer(in_chs=64, im_size=int(im_size/8), depth=6)



    def forward(self, x):
        pass
