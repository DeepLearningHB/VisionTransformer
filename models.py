import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from torch import Tensor
def practice():
    patch_size = 16
    in_channels = 3
    img_size = 224
    embedding_size = 768

    test = torch.randn(8, 3, 224, 224)

    projection = nn.Sequential(
        nn.Conv2d(in_channels, embedding_size, kernel_size=patch_size, stride=patch_size), # (b, emb, 14, 14)
        Rearrange('b e (h) (w) -> b (h w) e') # (b * (14 * 14) * emb)
    )

    projected_x = projection(test)

    cls_tokens = nn.Parameter(torch.randn(1, 1, embedding_size)) # 시작 토큰
    positions = nn.Parameter(torch.randn((img_size // patch_size ** 2 + 1, embedding_size))) # Learnable Positional Encoding
    cls_tokens = repeat(cls_tokens, '() n e -> b n e', b=8)

    cat_x = torch.cat([cls_tokens, projected_x], dim=1)
    print(cat_x.shape)

    cat_x += positions
    #------------------------------ patch embedding
    k = nn.Linear(embedding_size, embedding_size)
    q =nn.Linear(embedding_size, embedding_size)
    v = nn.Linear(embedding_size, embedding_size)
    num_heads = 8

    k = rearrange(k(cat_x), 'b n (h d) -> b h n d', h=num_heads) # (b p*p emb) -> (b, head, p*p, emb/head)
    q = rearrange(q(cat_x), 'b n (h d) -> b h n d', h=num_heads) # (b p*p emb) -> (b, head, p*p, emb/head)
    v = rearrange(v(cat_x), 'b n (h d) -> b h n d', h=num_heads) # (b p*p emb) -> (b, head, p*p, emb/head)

    #---------------------------------Multi head attention.

    print(k.shape)

    energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)
    scaling_factor = embedding_size ** 0.5
    att = F.softmax(energy / scaling_factor, dim=-1)

    out = torch.einsum('bhal, bhlv -> bhav', att, v)
    out = rearrange(out, 'b h n d -> b n (h d)')


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        return x + res

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_size = emb_size

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, self.emb_size, kernel_size=patch_size, stride=patch_size),  # (b, emb, 14, 14)
            Rearrange('b e (h) (w) -> b (h w) e')  # (b * (14 * 14) * emb)
        )

        self.cls_tokens = nn.Parameter(torch.randn(1, 1, self.emb_size))  # 시작 토큰
        self.positions = nn.Parameter(
            torch.randn((img_size // patch_size ** 2 + 1, self.emb_size)))  # Learnable Positional Encoding

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        print(x.shape)
        cls_tokens = repeat(self.cls_tokens, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        print(x.shape)
        print(self.positions.shape)
        x += self.positions
        return x # (batch, patch * patch, embedding)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=768, drop_p=0., forward_expansion=4, forward_drop_p=0., **kwargs):
        super(TransformerEncoderBlock, self).__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p)
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p)
                )
            )
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))

class ViT(nn.Sequential):
    def __init__(self,
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

if __name__ == "__main__":
    x = torch.randn(8, 3, 32, 32)
    model = ViT(patch_size=4, emb_size=256, img_size=32, depth=8, n_classes=10)
    print(model(x).shape)