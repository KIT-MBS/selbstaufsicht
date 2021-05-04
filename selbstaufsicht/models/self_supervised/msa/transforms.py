from ...utils import rna_to_index

class MSADataTransform(object):
    """
    Transforms for multiple sequence alignment data

    Transform::
        String of RNA letters to token indices
    """

    def __init__(self, ):
        return

    def __call__(self, msa):
        return [[rna_to_index[a] for a in sequence] for sequence in msa]
