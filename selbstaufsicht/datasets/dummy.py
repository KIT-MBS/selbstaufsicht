from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from .shrinked_force_permutations import ShrinkedForcePermutationsDataset
from ..utils import rna2index

class DummyDataset(ShrinkedForcePermutationsDataset):
    def __init__(self, transform=None):
        super().__init__()
        self.token_mapping = rna2index
        self.transform = transform
        self.samples = [MultipleSeqAlignment([
                        SeqRecord(Seq("AAAACCCC"), id='eins'),
                        SeqRecord(Seq("AAAACCCC"), id='zwei'),
                        SeqRecord(Seq("AAAACCCC"), id='drei'),
                        ]),
                        MultipleSeqAlignment([
                            SeqRecord(Seq("AAAAAAAC"), id='eins'),
                            SeqRecord(Seq("AAAAAAAU"), id='zwei'),
                            SeqRecord(Seq("AAAAAAAG"), id='drei'),
                        ]),
                        MultipleSeqAlignment([
                            SeqRecord(Seq("AAAAAAAC"), id='eins'),
                            SeqRecord(Seq("--AAAA--"), id='zwei'),
                            SeqRecord(Seq("---AAAA-"), id='drei'),
                        ]),
                        MultipleSeqAlignment([
                            SeqRecord(Seq("AAAA--------CAUA"), id='eins'),
                            SeqRecord(Seq("AAAA--------CAGA"), id='zwei'),
                            SeqRecord(Seq("AAAA--------CA.A"), id='drei'),
                        ]),
                        ]