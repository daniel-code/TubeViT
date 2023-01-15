import numpy as np
from torch import Tensor
from torchvision.models import ViT_B_16_Weights

from TubeViT.model import TubeViT

if __name__ == '__main__':
    num_classes = 1
    batch_size = 2
    frames_per_clip = 32

    x = np.random.random((batch_size, 3, frames_per_clip, 224, 224))
    x = Tensor(x)
    print('x: ', x.shape)

    y = np.random.randint(0, 1, size=(batch_size, num_classes))
    y = Tensor(y)
    print('y: ', y.shape)

    model = TubeViT(num_classes=num_classes,
                    video_shape=x.shape[1:],
                    num_layers=12,
                    num_heads=12,
                    hidden_dim=768,
                    mlp_dim=3072)

    weights = ViT_B_16_Weights.DEFAULT.get_state_dict(progress=True)
    # remove missmatch parameters
    weights.pop('encoder.pos_embedding')
    weights.pop('heads.head.weight')
    weights.pop('heads.head.bias')

    model.load_state_dict(weights, strict=False)
    print(model)
