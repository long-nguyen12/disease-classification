import torch
from torch import nn, Tensor
from models.layers import DropPath
from models.modules.common import ECA


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping"""

    def __init__(self, patch_size=16, stride=16, padding=0, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride, padding)

    def forward(self, x: torch.Tensor) -> Tensor:
        x = self.proj(x)  # b x hidden_dim x 14 x 14
        return x


class Pooling(nn.Module):
    def __init__(self, pool_size=3) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, 1, pool_size // 2, count_include_pad=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x) - x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PoolFormerBlock(nn.Module):
    def __init__(self, dim, pool_size=3, dpr=0.0, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.token_mixer = Pooling(pool_size)
        self.norm2 = nn.GroupNorm(1, dim)
        self.drop_path = DropPath(dpr) if dpr > 0.0 else nn.Identity()
        self.mlp = MLP(dim, int(dim * 4))

        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
            * self.token_mixer(self.norm1(x))
        )
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        )
        return x


poolformer_settings = {
    "S24": [
        [4, 4, 4, 4],
        [64, 128, 320, 512],
        0.1,
    ],  # [layers, embed_dims, drop_path_rate]
    "S36": [[6, 6, 18, 6], [64, 128, 320, 512], 0.2],
    "M36": [[6, 6, 18, 6], [96, 192, 384, 768], 0.3],
}


class PoolFormer(
    nn.Module
):  # this model works with any image size, even non-square image size
    def __init__(
        self,
        model_name: str = "S24",
        pretrained: str = None,
        num_classes: int = 1000,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        assert (
            model_name in poolformer_settings.keys()
        ), f"PoolFormer model name should be in {list(poolformer_settings.keys())}"
        layers, embed_dims, drop_path_rate = poolformer_settings[model_name]
        self.layers = layers
        self.patch_embed = PatchEmbed(7, 4, 2, 3, embed_dims[0])

        network = []
        for i in range(len(layers)):
            blocks = []
            for j in range(layers[i]):
                dpr = drop_path_rate * (j + sum(layers[:i])) / (sum(layers) - 1)
                blocks.append(PoolFormerBlock(embed_dims[i], 3, dpr))

            network.append(nn.Sequential(*blocks))

        patch_embeds = []
        for i in range(len(layers)):
            if i >= len(layers) - 1:
                break
            patch_embeds.append(PatchEmbed(3, 2, 1, embed_dims[i], embed_dims[i + 1]))

        classifier_blocks = [3, 2, 1, 0]
        self.classifiers = nn.ModuleList([])
        for i, embed_dim in enumerate(embed_dims):
            # if i >= len(layers) - 1:
            #     self.classifiers.append(
            #         nn.Sequential(
            #             nn.GroupNorm(1, embed_dim), nn.Linear(embed_dim, num_classes)
            #         )
            #     )
            # else:
            #     self.classifiers.append(
            #         ClassifierModule(classifier_blocks[i], embed_dim, num_classes)
            #     )
            self.classifiers.append(
                ClassifierModule(classifier_blocks[i], embed_dim, num_classes)
            )

        self.network = nn.ModuleList(network)
        self.patch_embeds = nn.ModuleList(patch_embeds)

        # self.norm = nn.GroupNorm(1, embed_dims[-1])
        # self.head = nn.Linear(embed_dims[-1], num_classes)

        self._init_weights(pretrained)

        # self.fork_feat(embed_dims)

    def fork_feat(self, embed_dims):
        self.out_indices = [0, 2, 4, 6]

        for i, index in enumerate(self.out_indices):
            self.add_module(f"norm{index}", nn.GroupNorm(1, embed_dims[i]))

    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            try:
                self.load_state_dict(
                    torch.load(pretrained, map_location="cpu"), strict=False
                )
            except RuntimeError:
                pretrained_dict = torch.load(pretrained, map_location="cpu")
                pretrained_dict.popitem()  # remove bias
                pretrained_dict.popitem()  # remove weight
                self.load_state_dict(pretrained_dict, strict=False)
            finally:
                print(f"Loaded imagenet pretrained from {pretrained}")
        else:
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    if n.startswith("head"):
                        nn.init.zeros_(m.weight)
                        nn.init.zeros_(m.bias)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def return_features(self, x: Tensor):
        x = self.patch_embed(x)
        outs = []

        for i, blk in enumerate(self.network):
            x = blk(x)

            if i in self.out_indices:
                out = getattr(self, f"norm{i}")(x)
                outs.append(out)
        return outs

    def forward(self, x: Tensor):
        x = self.patch_embed(x)
        outs = []
        for i, blk in enumerate(self.network):
            x = blk(x)
            outs.append(self.classifiers[i](x))
            if i >= len(self.layers) - 1:
                break
            x = self.patch_embeds[i](x)

        return outs


class ClassifierModule(nn.Module):
    def __init__(self, num_blocks, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.num_blocks = num_blocks
        attentions = []

        for i in range(num_blocks):
            attentions.append(ECA(channel))

        self.attentions = nn.ModuleList(attentions)
        self.norm = nn.GroupNorm(1, channel)
        self.linear = nn.Linear(channel, num_classes)

    def forward(self, x):
        res = x
        for i in range(len(self.attentions)):
            res = self.attentions[i](res)

        res = self.norm(res)
        res = res.mean([-2, -1])
        return self.linear(res)


if __name__ == "__main__":
    model = PoolFormer("S24")
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(len(y))
