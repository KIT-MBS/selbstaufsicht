from typing import Dict, Union
import torch
import torch.nn as nn


class InpaintingHead(nn.Module):
    def __init__(self, d: int, num_classes: int, device: Union[str, torch.device] = None, dtype: torch.dtype = None, boot: bool = False) -> None:
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
        self.boot = boot

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
        if not self.boot:
            output = output[x['mask']]

        output = output.reshape(-1, self.num_classes)  # [B*E*L, NCLasses]
        return output


class JigsawHead(nn.Module):
    def __init__(self,
                 d: int,
                 num_classes: int,
                 proj_linear: bool = True,
                 euclid_emb: bool = False,
                 layer_norm_eps: float = 1e-5,
                 device: Union[str, torch.device] = None,
                 dtype: torch.dtype = None, boot: bool = False, frozen: bool = False,
                 seq_dist: bool = False) -> None:
        """
        Initializes the head module for the jigsaw upstream task.

        Args:
            d (int): Embedding dimensionality.
            num_classes (int): Number of classes (number of allowed permutations)
            proj_linear (bool): if True uses a linear projection head, if False uses two layers with LayerNorm and ReLU, Defaults to True.
            euclid_emb (bool, optional): Whether Euclidean embedding of the discrete permutation metric is used. Defaults to False.
            layer_norm_eps (float, optional): Epsilon used by LayerNormalization. Defaults to 1e-5.
            device (Union[str, torch.device], optional): Used computation device. Defaults to None.
            dtype (torch.dtype, optional): Used tensor dtype. Defaults to None.
        """

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(JigsawHead, self).__init__()
        self.euclid_emb = euclid_emb
        if proj_linear:
            self.proj = nn.Linear(d, num_classes, **factory_kwargs)
        else:
            self.proj = nn.Sequential(
                nn.Linear(d, d, **factory_kwargs),
                nn.LayerNorm(d, eps=layer_norm_eps, **factory_kwargs),
                nn.ReLU(),
                nn.Linear(d, num_classes, bias=False, **factory_kwargs),
            )
        self.num_classes = num_classes
        self.boot = boot
        self.frozen = frozen
        self.seq_dist = seq_dist

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
        if self.frozen:
            latent = latent[:,0,0,:] #[B,D]
        else:
            latent = latent[:, :, 0, :]  # [B, E, D]
        
        if self.euclid_emb:
            return self.proj(latent)  # [B, E, NClasses]
        else:
            output = self.proj(latent)
            if self.boot:
                if self.seq_dist:
                    return output.reshape(output.shape[1],)
                else:
                    return output.reshape(-1, self.num_classes)
            else:
                if self.frozen:
                    return self.proj(latent)
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


class ContactHead(nn.Module):
    def __init__(self, num_maps, cull_tokens, log_clamp_min: float = 1e-6, num_layers=1, activation='sigmoid', num_hidden=None, bias=False, device: Union[str, torch.device] = None, dtype: torch.dtype = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ContactHead, self).__init__()
        self.num_maps = num_maps
        if num_hidden is None:
            num_hidden = 2*num_maps
        self.cull_tokens = cull_tokens
        self.log_clamp_min = log_clamp_min
        if num_layers == 1:
            self.proj = nn.Conv2d(num_maps, 1, kernel_size=1, bias=bias, **factory_kwargs)
        elif num_layers >= 2:
            layers = [nn.Conv2d(num_maps, num_hidden, kernel_size=1, bias=bias, **factory_kwargs), get_activation(activation)]
            for i in range(num_layers-2):
                layers.append(nn.Conv2d(num_hidden, num_hidden, kernel_size=1, bias=bias, **factory_kwargs))
                layers.append(get_activation(activation))
            layers.append(nn.Conv2d(num_hidden, 1, kernel_size=1, bias=bias, **factory_kwargs))

            self.proj = nn.Sequential(*layers)
        else:
            raise ValueError(f"Expected num_layers >= 1, got {num_layers}")

    def forward(self, latent, x) -> torch.Tensor:
        # TODO only tied axial attention for now
        # TODO only batch size 1 for now
        # TODO remove start seq, delimiter, and mask tokens in one go
        # TODO think about when delimiter tokens are not aligned
        B, E, L = x['msa'].size()
        assert B == 1

        mask = torch.ones((L, ), device=x['msa'].device)
        for token in self.cull_tokens:
            mask -= (x['msa'][:, 0].reshape((L, )) == token).int()
        mask = mask.bool()
        degapped_L = int(mask.sum())
        mask = torch.reshape(mask, (B, 1, L))
        mask = torch.logical_and(mask.reshape((B, 1, L)).expand(B, L, L), mask.reshape((B, L, 1)).expand(B, L, L))
        mask = mask.reshape((B, 1, L, L)).expand((B, self.num_maps, L, L))

        out = x['attn_maps']
        out = torch.cat([m.squeeze(dim=2) for m in out], dim=1)  # [B, num_blocks * H, L, L]
        out = out.masked_select(mask).reshape(B, self.num_maps, degapped_L, degapped_L)
        out = self.proj(out)  # [B, 1, L, L]
        out = (out + torch.transpose(out, -1, -2)) * 0.5
        # NOTE Sigmoid is symmetrical
        out = torch.cat((-out, out), dim=1)  # [B, 2, L, L]

        return out


def get_activation(s):
    if s == 'sigmoid':
        return nn.Sigmoid()
    elif s == 'relu':
        return nn.ReLU
    raise ValueError()
