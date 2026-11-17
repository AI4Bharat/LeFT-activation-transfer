# Target configurations

Four architectures, one transferred curve, four datasets and five seeds, giving
the 16 architecture × dataset combinations of the main table.

| Config            | Architecture                     | Attention        | Params |
| ----------------- | -------------------------------- | ---------------- | ------ |
| `vit_d2.yaml`     | ViT, depth 2, embed 64           | softmax, global  | ~114K  |
| `swin_d2.yaml`    | Swin, depth 2, window 4, patch 4 | shifted-window   | ~114K  |
| `linvit_d2.yaml`  | ViT with elu+1 linear attention  | linear kernel    | ~114K  |
| `mlp_mixer.yaml`  | MLP-Mixer, depth 2, embed 76     | none             | ~130K  |

`mlp_ratio`, `num_heads` and `dropout` are not tuned per architecture; their
values hold the four architectures within a single parameter band, which is the
condition under which a margin is comparable across rows.
