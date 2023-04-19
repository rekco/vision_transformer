import torch
from torch import nn, Tensor
from ViT import Transformer


class EncoderBlock(nn.Module):
    def __init__(self,
                 in_chs: int = 3,
                 im_size: int = 224,
                 num_blocks: int = 1,
                 block_depth: int = 1,
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
                 in_chs: int,
                 out_chs: int):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_chs)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_chs)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_chs, out_chs, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_chs)
        self.relu3 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x


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


class PreProcess(nn.Sequential):
    def __init__(self, in_chs: int, out_chs: int):
        super(PreProcess, self).__init__(
            nn.Conv2d(in_chs, out_chs, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
            nn.Conv2d(out_chs, out_chs, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chs),
            nn.ReLU()
        )


class Output(nn.Sequential):
    def __init__(self, in_chs, out_chs):
        super(Output, self).__init__(
            nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(),
            nn.Conv2d(in_chs, in_chs, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(),
            nn.Conv2d(in_chs, out_chs, kernel_size=1),
            nn.BatchNorm2d(out_chs),
            nn.Sigmoid()
        )


class UFormer(nn.Module):
    def __init__(self,
                 in_chs: int = 3,
                 out_chs: int = 1,
                 im_size: int = 224):
        super(UFormer, self).__init__()

        self.preprocess = PreProcess(in_chs, 16)
        self.encoder1 = EncoderBlock(in_chs=16, im_size=im_size, num_blocks=2)

        self.down_sample1 = DownSample(16, 32)
        self.encoder2 = EncoderBlock(in_chs=32, im_size=int(im_size/2), num_blocks=2)

        self.down_sample2 = DownSample(32, 64)
        self.encoder3 = EncoderBlock(in_chs=64, im_size=int(im_size/4), num_blocks=6)

        self.down_sample3 = DownSample(64,  128)
        self.bottleneck = Transformer(in_chs=128, im_size=int(im_size/8), depth=6)

        self.up_sample3 = UpSample(128, 64)
        self.decoder3 = DecoderBlock(128, 64)

        self.up_sample2 = UpSample(64, 32)
        self.decoder2 = DecoderBlock(64, 32)

        self.up_sample1 = UpSample(32, 16)
        self.decoder1 = DecoderBlock(32, 16)

        self.output = Output(16, out_chs)

    def forward(self, x: Tensor):
        pre_feats = self.preprocess(x)
        en_feats1 = self.encoder1(pre_feats)

        en_feats2 = self.down_sample1(en_feats1)
        en_feats2 = self.encoder2(en_feats2)

        en_feats3 = self.down_sample2(en_feats2)
        en_feats3 = self.encoder3(en_feats3)

        en_feats4 = self.down_sample3(en_feats3)
        en_feats4 = self.bottleneck(en_feats4)

        up_feats3 = self.up_sample3(en_feats4)
        up_feats3 = torch.cat([up_feats3, en_feats3], dim=1)
        up_feats3 = self.decoder3(up_feats3)

        up_feats2 = self.up_sample2(up_feats3)
        up_feats2 = torch.cat([up_feats2, en_feats2], dim=1)
        up_feats2 = self.decoder2(up_feats2)

        up_feats1 = self.up_sample1(up_feats2)
        up_feats1 = torch.cat([up_feats1, en_feats1], dim=1)
        up_feats1 = self.decoder1(up_feats1)

        output = self.output(up_feats1)

        return output


if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224).cuda()
    m = UFormer().cuda()
    y = m(x)
    print(y.shape)

