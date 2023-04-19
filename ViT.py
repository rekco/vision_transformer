import torch
from torch import nn, Tensor
from einops import rearrange
from einops.layers.torch import Rearrange


class PositionEmbedding(nn.Module):
    def __init__(self,
                 in_chs: int,
                 im_size: int,
                 patch_size: int = 16,
                 emb_size: int = 1024,
                 ):
        super(PositionEmbedding, self).__init__()
        self.patch_size = patch_size
        self.project = nn.Sequential(
            # nn.Conv2d(in_chs, emb_size, kernel_size=(patch_size, patch_size), stride=patch_size),
            # Rearrange('b e h w -> b (h w) e')
            Rearrange('b c (h s1) (w s2) -> b (h w) (c s1 s2)', s1=patch_size, s2=patch_size),
            nn.Linear(in_chs * patch_size ** 2, emb_size)
        )
        self.positions = nn.Parameter(torch.randn((im_size // patch_size)**2, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.project(x)
        x += self.positions

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 emb_size: int = 1024,
                 num_heads: int = 8,
                 dropout: float = 0):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.att_out = nn.Dropout(dropout)
        self.project = nn.Linear(emb_size, emb_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        q = rearrange(self.query(x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.key(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.value(x), 'b n (h d) -> b h n d', h=self.num_heads)

        scale = self.emb_size ** 0.5
        qk = torch.einsum('b h q d, b h k d -> b h q k', q, k)
        att = self.softmax(qk) / scale
        att = torch.einsum('b h i j, b h j k -> b h i k', att, v)
        att = rearrange(att, 'b h n d -> b n (h d)')
        att = self.project(att)

        return att


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.fn(x)
        x += res

        return x


class FFN(nn.Sequential):
    def __init__(self,
                 emb_size: int = 1024,
                 expansion: int = 4,
                 drop_p: float = 0.):
        super(FFN, self).__init__(
            nn.Linear(emb_size, emb_size * expansion),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(emb_size * expansion, emb_size)
        )


class TransformerBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 1024,
                 num_head: int = 8,
                 drop_p: float = 0.,
                 ffn_expansion: int = 4,
                 ffn_drop_p: float = 0.):
        super(TransformerBlock, self).__init__(
            Residual(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_head, drop_p),
                nn.Dropout(drop_p)
            )),
            Residual(nn.Sequential(
                nn.LayerNorm(emb_size),
                FFN(emb_size, ffn_expansion, ffn_drop_p),
                nn.Dropout(ffn_drop_p)
            ))
        )


class Patches2Im(nn.Module):
    def __init__(self, im_size: int = 224, in_chs: int = 3, patch_size: int = 16, emb_size: int = 1024):
        super(Patches2Im, self).__init__()
        self.h = im_size // patch_size
        self.w = im_size // patch_size
        self.s1 = patch_size
        self.s2 = patch_size
        self.project = nn.Linear(emb_size, in_chs * patch_size ** 2)

    def forward(self, x) -> Tensor:
        x = self.project(x)
        x = rearrange(x, 'b (h w) (s1 s2 c) -> b c (h s1) (w s2)', h=self.h, w=self.w, s1=self.s1, s2=self.s2)

        return x


class LocalFormer(nn.Module):
    def __init__(self,
                 in_chs: int = 3,
                 im_size: int = 224,
                 depth: int = 2,
                 patch_size: int = 16,
                 emb_size: int = 1024,
                 num_heads: int = 8,
                 att_drop_p: float = 0.,
                 ffn_expansion: int = 4,
                 ffn_drop_p: float = 0.):
        super(LocalFormer, self).__init__()
        self.position_embedding = DilatePositionEmbedding(in_chs, im_size, patch_size, emb_size)
        self.transformers = nn.Sequential(*[TransformerBlock(emb_size, num_heads, att_drop_p, ffn_expansion, ffn_drop_p) for i in range(depth)])
        self.patch_im = Patches2Im(im_size, in_chs, patch_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.position_embedding(x)
        for transformer in self.transformers:
            x = transformer(x)
        x = self.patch_im(x)

        return x
