from typing import Dict, Union
import torch
import torch.nn as nn


class InpaintingHead(nn.Module):
    def __init__(self, d: int, num_classes: int, device: Union[str, torch.device] = None, dtype: torch.dtype = None) -> None:
        """
        Initializes the head module for the inpaiting upstream task.

        Args:
            d (int): Embedding dimensionality.
            num_classes (int): Number of classes (number of predictable tokens)
            device (Union[str, torch.device], optional): Used computation device. Defaults to None.
            dtype (torch.dtype, optional): Used tensor dtype. Defaults to None.
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(InpaintingHead, self).__init__()
        # TODO assuming the last layer in an encoder block is a nonlinearity
        self.num_classes = num_classes
        self.proj = nn.Linear(d, num_classes, **factory_kwargs)

    # NOTE the output is basically flattened (of  shape (-1, num_classes)) since the number of masked tokens per sample in the batch is not the same
    def forward(self, latent: torch.Tensor, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Receives latent representation, performs linear transformation and applies inpainting mask to predict masked tokens.

        Args:
            latent (torch.Tensor): Latent representation [B, E, L, D].
            x (Dict[str, torch.Tensor]): Input data.

        Returns:
            torch.Tensor: Inpainting prediction [B*E*L, NClasses].
        """

        # latent is of shape [B, E, L, D]
        output = self.proj(latent)  # [B, E, L, NClasses]
        output = output[x['mask']]

        output = output.reshape(-1, self.num_classes)  # [B*E*L, NCLasses]
        return output


class JigsawHead(nn.Module):
    def __init__(self,
                 d: int,
                 num_classes: int,
                 proj_linear: bool = True,
                 euclid_emb: torch.Tensor = None,
                 layer_norm_eps: float = 1e-5,
                 device: Union[str, torch.device] = None,
                 dtype: torch.dtype = None) -> None:
        """
        Initializes the head module for the jigsaw upstream task.

        Args:
            d (int): Embedding dimensionality.
            num_classes (int): Number of classes (number of allowed permutations)
            proj_linear (bool): if True uses a linear projection head, if False uses two layers with LayerNorm and ReLU, Defaults to True.
            euclid_emb (torch.Tensor): Euclidean embedding of the discrete permutation metric. Defaults to None.
            layer_norm_eps (float, optional): Epsilon used by LayerNormalization. Defaults to 1e-5.
            device (Union[str, torch.device], optional): Used computation device. Defaults to None.
            dtype (torch.dtype, optional): Used tensor dtype. Defaults to None.
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(JigsawHead, self).__init__()
        self.euclid_emb = euclid_emb
        if self.euclid_emb is not None:
            self.euclid_emb = self.euclid_emb.to(device)
        if proj_linear:
            self.proj = nn.Linear(d, num_classes, **factory_kwargs)
        else:
            self.proj = nn.Sequential(
                nn.Linear(d, d, **factory_kwargs),
                nn.LayerNorm(d, eps=layer_norm_eps, **factory_kwargs),
                nn.ReLU(),
                nn.Linear(d, num_classes, bias=False, **factory_kwargs),
            )

    def forward(self, latent: torch.Tensor, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Receives latent representation, performs linear transformation to predict applied permutations.

        Args:
            latent (torch.Tensor): Latent representation [B, E, L, D].
            x (Dict[str, torch.Tensor]): Input data.

        Returns:
            torch.Tensor: Jigsaw prediction [B, NClasses, E].
        """

        # latent is of shape [B, E, L, D]
        latent = latent[:, :, 0, :]  # [B, E, D]
        if self.euclid_emb is not None:
            return self.euclid_emb[torch.argmax(self.proj(latent), dim=-1), :]  # [B, E, M]
        else:
            return torch.transpose(self.proj(latent), 1, 2)  # [B, NClasses, E]


# TODO different hidden and out dim?
class ContrastiveHead(nn.Module):
    def __init__(self, d: int, layer_norm_eps: float = 1e-5, device: Union[str, torch.device] = None, dtype: torch.dtype = None) -> None:
        """
        Initializes the head module for the contrastive upstream task.

        Args:
            d (int): Embedding dimensionality.
            layer_norm_eps (float, optional): Epsilon used by LayerNormalization. Defaults to 1e-5.
            device (Union[str, torch.device], optional): Used computation device. Defaults to None.
            dtype (torch.dtype, optional): Used tensor dtype. Defaults to None.
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ContrastiveHead, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(d, d, **factory_kwargs),
            nn.LayerNorm(d, eps=layer_norm_eps, **factory_kwargs),
            nn.ReLU(),
            nn.Linear(d, d, bias=False, **factory_kwargs),
        )

    def forward(self, latent: torch.Tensor, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Receives latent representation, passes mean over MSA through 2-layer-Feedforward-NN to create another latent representation,
        which is then to be compared to that of the contrastive input.

        Args:
            latent (torch.Tensor): Latent representation [B, E, L, D].
            x (Dict[str, torch.Tensor]): Input data.

        Returns:
            torch.Tensor: Contrastive latent representation [B, D].
        """

        # latent is of shape [B, E, L, D]
        latent = latent.mean(dim=-2)  # [B, E, D]
        latent = latent.mean(dim=-2)  # [B, D]

        return self.proj(latent)
