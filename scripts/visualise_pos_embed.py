import matplotlib.pyplot as plt
import torch

from tubevit.positional_encoding import get_3d_sincos_pos_embed

if __name__ == "__main__":
    kernel_sizes = (
        (8, 8, 8),
        (16, 4, 4),
        (4, 12, 12),
        (1, 16, 16),
    )

    strides = (
        (16, 32, 32),
        (6, 32, 32),
        (16, 32, 32),
        (32, 16, 16),
    )

    offsets = (
        (0, 0, 0),
        (4, 8, 8),
        (0, 16, 16),
        (0, 0, 0),
    )

    tube_shape = (
        (2, 7, 7),
        (3, 7, 7),
        (2, 7, 7),
        (1, 14, 14),
    )

    pos_encode = [torch.zeros(1, 768)]

    for i in range(len(kernel_sizes)):
        pos_embed = get_3d_sincos_pos_embed(
            embed_dim=768, tube_shape=tube_shape[i], stride=strides[i], offset=offsets[i], kernel_size=kernel_sizes[i]
        )
        # pos_embed = torch.Tensor(pos_embed)
        pos_encode.append(pos_embed)

    pos_encode = torch.cat(pos_encode)
    plt.imshow(pos_encode)
    plt.title("Position Embedding")
    plt.xlabel("embed_dim")
    plt.ylabel("index of tokens")
    plt.tight_layout()
    plt.savefig("Position_Embedding.png")
    plt.show()
