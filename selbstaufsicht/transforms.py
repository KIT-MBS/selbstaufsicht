from typing import Dict, List, Tuple
from Bio.Align import MultipleSeqAlignment
import torch

class SelfSupervisedCompose:
    """
    Composes several transforms together.
    similar to torchvision.transforms.Compose.
    """

    def __init__(self, transforms: List[object]) -> None:
        """
        Initializes SelfSupervisedCompose.

        Args:
            transforms (List[object]): List of preprocessing transforms.
        """
        
        self.transforms = transforms

    def __call__(self, sample: Dict[str, MultipleSeqAlignment], target: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Performs the given transforms in the specified order.

        Args:
            sample (Dict[str, MultipleSeqAlignment]): Lettered MSA.
            target (Dict[str, torch.Tensor]): Upstream task labels.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: x: Tokenized MSA, y: Upstream task labels.
        """
        
        for t in self.transforms:
            sample, target = t(sample, target)
        return (sample, target)

    def __repr__(self) -> str:
        """
        Returns string representation of the transform composition.

        Returns:
            str: String representation of the transform composition.
        """
        
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
