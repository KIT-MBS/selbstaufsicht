from copy import deepcopy
import torch
import torch.testing as testing

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.transforms import RandomMSAMasking, ExplicitPositionalEncoding, RandomMSACropping, RandomMSASubsampling, RandomMSAShuffling
from selbstaufsicht.models.self_supervised.msa.utils import MSACollator, _pad_collate_nd


def test_msa_tokenize(tokenized_msa):
    tokenized_msa_ref = torch.tensor([[17, 3, 5, 4, 5, 5, 4, 3],
                                      [17, 3, 3, 4, 1, 5, 4, 3],
                                      [17, 5, 5, 4, 3, 5, 4, 1],
                                      [17, 4, 5, 4, 5, 5, 4, 5]])
    testing.assert_close(tokenized_msa['msa'], tokenized_msa_ref, rtol=0, atol=0)


def test_msa_mask_token(tokenized_msa):
    masking = RandomMSAMasking(
        p=.5, mode='token', mask_token=rna2index['MASK_TOKEN'])
    positional = ExplicitPositionalEncoding()

    x, y = masking(tokenized_msa)
    x, y = positional((x, y))

    x_ref = {'aux_features': torch.tensor([[[0.0000, 0.0000],
                                            [0.0010, 0.1250],
                                            [0.0020, 0.2500],
                                            [0.0030, 0.3750],
                                            [0.0040, 0.5000],
                                            [0.0050, 0.6250],
                                            [0.0060, 0.7500],
                                            [0.0070, 0.8750]]]),
             'msa': torch.tensor([[17,  3, 19,  4, 19,  5, 19,  3],
                                  [17, 19,  3,  4,  1,  5,  4, 19],
                                  [17,  5, 19,  4, 19, 19, 19,  1],
                                  [17, 19, 19, 19,  5, 19,  4, 19]]),
             'mask': torch.tensor([[False, False,  True, False,  True, False,  True, False],
                                   [False,  True, False, False, False, False, False,  True],
                                   [False, False,  True, False,  True,  True,  True, False],
                                   [False,  True,  True,  True, False,  True, False,  True]])}

    y_ref = {'inpainting': torch.tensor(
        [5, 5, 4, 3, 3, 5, 3, 5, 4, 4, 5, 4, 5, 5])}

    testing.assert_close(
        x['aux_features'], x_ref['aux_features'], atol=1e-4, rtol=1e-3)
    testing.assert_close(x['msa'], x_ref['msa'], rtol=0, atol=0)
    testing.assert_close(x['mask'], x_ref['mask'], rtol=0, atol=0)
    testing.assert_close(y['inpainting'], y_ref['inpainting'], rtol=0, atol=0)


def test_msa_mask_column(tokenized_msa):
    masking = RandomMSAMasking(
        p=.5, mode='column', mask_token=rna2index['MASK_TOKEN'])
    positional = ExplicitPositionalEncoding()

    x, y = masking(tokenized_msa)
    x, y = positional((x, y))

    x_ref = {'aux_features': torch.tensor([[[0.0000, 0.0000],
                                            [0.0010, 0.1250],
                                            [0.0020, 0.2500],
                                            [0.0030, 0.3750],
                                            [0.0040, 0.5000],
                                            [0.0050, 0.6250],
                                            [0.0060, 0.7500],
                                            [0.0070, 0.8750]]]),
             'msa': torch.tensor([[17, 19, 5, 19, 5, 19, 4, 19],
                                  [17, 19, 3, 19, 1, 19, 4, 19],
                                  [17, 19, 5, 19, 3, 19, 4, 19],
                                  [17, 19, 5, 19, 5, 19, 4, 19]]),
             'mask': torch.tensor([[False, True, False, True, False, True, False, True],
                                   [False, True, False, True, False, True, False, True],
                                   [False, True, False, True, False, True, False, True],
                                   [False, True, False, True, False, True, False, True]])}

    y_ref = {'inpainting': torch.tensor(
        [3, 4, 5, 3, 3, 4, 5, 3, 5, 4, 5, 1, 4, 4, 5, 5])}

    testing.assert_close(
        x['aux_features'], x_ref['aux_features'], atol=1e-4, rtol=1e-3)
    testing.assert_close(x['msa'], x_ref['msa'], rtol=0, atol=0)
    testing.assert_close(x['mask'], x_ref['mask'], rtol=0, atol=0)
    testing.assert_close(y['inpainting'], y_ref['inpainting'], rtol=0, atol=0)


def test_msa_mask_block(tokenized_msa):
    masking = RandomMSAMasking(
        p=.5, mode='block', mask_token=rna2index['MASK_TOKEN'])
    positional = ExplicitPositionalEncoding()

    x, y = masking(tokenized_msa)
    x, y = positional((x, y))

    x_ref = {'aux_features': torch.tensor([[[0.0000, 0.0000],
                                            [0.0010, 0.1250],
                                            [0.0020, 0.2500],
                                            [0.0030, 0.3750],
                                            [0.0040, 0.5000],
                                            [0.0050, 0.6250],
                                            [0.0060, 0.7500],
                                            [0.0070, 0.8750]]]),
             'msa': torch.tensor([[17, 3, 5, 19, 19, 19, 4, 3],
                                  [17, 3, 3, 19, 19, 19, 4, 3],
                                  [17, 5, 5, 19, 19, 19, 4, 1],
                                  [17, 4, 5, 19, 19, 19, 4, 5]]),
             'mask': torch.tensor([[False, False, False, True, True, True, False, False],
                                   [False, False, False, True, True, True, False, False],
                                   [False, False, False, True, True, True, False, False],
                                   [False, False, False, True, True, True, False, False]])}

    y_ref = {'inpainting': torch.tensor(
        [4, 5, 5, 4, 1, 5, 4, 3, 5, 4, 5, 5])}

    testing.assert_close(
        x['aux_features'], x_ref['aux_features'], atol=1e-4, rtol=1e-3)
    testing.assert_close(x['msa'], x_ref['msa'], rtol=0, atol=0)
    testing.assert_close(x['mask'], x_ref['mask'], rtol=0, atol=0)
    testing.assert_close(y['inpainting'], y_ref['inpainting'], rtol=0, atol=0)


def test_jigsaw(tokenized_msa):
    permutations = torch.tensor([[0, 1],
                                 [1, 0]], dtype=torch.int64)
    label = torch.tensor([0, 1, 0, 1])
    shuffling = RandomMSAShuffling(permutations=permutations)
    x, y = shuffling(deepcopy(tokenized_msa), label=label)

    x_ref = torch.tensor([[17, 3, 5, 4, 5, 5, 4, 3],
                          [17, 1, 5, 4, 3, 3, 4, 3],
                          [17, 5, 5, 4, 3, 5, 4, 1],
                          [17, 5, 5, 4, 4, 5, 4, 5]])

    testing.assert_close(x['msa'], x_ref, rtol=0, atol=0)
    testing.assert_close(y['jigsaw'], label, rtol=0, atol=0)

    permutations = torch.tensor([[0, 1, 2],
                                 [1, 0, 2],
                                 [0, 2, 1],
                                 [2, 0, 1]], dtype=torch.int64)
    label = torch.tensor([3, 2, 1, 0])
    shuffling = RandomMSAShuffling(permutations=permutations)
    x, y = shuffling(deepcopy(tokenized_msa), label=label)

    x_ref = torch.tensor([[17, 5, 4, 3, 5, 4, 5, 3],
                          [17, 3, 3, 5, 4, 4, 1, 3],
                          [17, 4, 3, 5, 5, 5, 4, 1],
                          [17, 4, 5, 4, 5, 5, 4, 5]])

    testing.assert_close(x['msa'], x_ref, rtol=0, atol=0)
    testing.assert_close(y['jigsaw'], label, rtol=0, atol=0)


def test_subsampling(basic_msa):
    sampler = RandomMSASubsampling(3, False, 'uniform')
    sampled = sampler(basic_msa)['msa']

    sampled_ref = MultipleSeqAlignment(
        [
            SeqRecord(Seq("CCUACU."), id='seq3'),
            SeqRecord(Seq("UCUCCUC"), id='seq4'),
            SeqRecord(Seq("ACUCCUA"), id='seq1'),
        ]
    )

    for idx in range(len(sampled)):
        assert sampled[idx].seq == sampled_ref[idx].seq
        
        
    
def test_cropping(basic_msa):
    sampler = RandomMSASubsampling(4, False, 'uniform')
    sampled = sampler(basic_msa)
    cropper = RandomMSACropping(5)
    cropped = cropper(sampled)
    
    cropped_ref = MultipleSeqAlignment(
        [
            SeqRecord(Seq("ACUCC"), id='seq1'),
            SeqRecord(Seq("AAU.C"), id='seq2'),
            SeqRecord(Seq("CCUAC"), id='seq3'),
            SeqRecord(Seq("UCUCC"), id='seq4'),
        ]
    )
    
    for idx in range(len(cropped)):
        assert cropped['msa'][idx].seq == cropped_ref[idx].seq


def test_pad_collate_nd():
    # Test 2d
    shapes = [(4, 3), (3, 4)]
    batch = []
    for idx in range(len(shapes)):
        batch.append(torch.ones((shapes[idx])))
    out, padding_mask = _pad_collate_nd(batch, need_padding_mask=True)
    out_ref = torch.ones((len(shapes), 4, 4))
    out_ref[0, :, -1] = 0.
    out_ref[1, -1, :] = 0.
    padding_mask_ref = torch.zeros((len(shapes), 4, 4), dtype=torch.bool)
    padding_mask_ref[0, :, -1] = True
    padding_mask_ref[1, -1, :] = True
    
    testing.assert_close(out, out_ref, atol=0, rtol=0)
    testing.assert_close(padding_mask, padding_mask_ref, atol=0, rtol=0)

    # Test 3d
    shapes = [(6, 4, 2), (1, 3, 5)]
    batch = []
    for idx in range(len(shapes)):
        batch.append(torch.ones((shapes[idx])))
    out, padding_mask = _pad_collate_nd(batch, need_padding_mask=True)
    out_ref = torch.ones((len(shapes), 6, 4, 5))
    out_ref[0, :, :, -3:] = 0.
    out_ref[1, -5:, :, :] = 0.
    out_ref[1, :, -1:, :] = 0.
    padding_mask_ref = torch.zeros((len(shapes), 6, 4, 5), dtype=torch.bool)
    padding_mask_ref[0, :, :, -3:] = True
    padding_mask_ref[1, -5:, :, :] = True
    padding_mask_ref[1, :, -1:, :] = True
    testing.assert_close(out, out_ref, atol=0, rtol=0)
    testing.assert_close(padding_mask, padding_mask_ref, atol=0, rtol=0)


def test_msa_collator():
    collator = MSACollator(0)
    B = 2
    S = [5, 4]
    L = [6, 7]
    inpainting = [3, 2]
    jigsaw = S
    contrastive_S = S
    contrastive_L = [8, 5]

    data = [({'msa': torch.zeros((S[idx], L[idx]), dtype=torch.int),
              'mask': torch.zeros((S[idx], L[idx]), dtype=torch.bool),
              'aux_features': torch.zeros((1, L[idx], 2)),
              'contrastive': torch.zeros((contrastive_S[idx], contrastive_L[idx]), dtype=torch.int),
              'aux_features_contrastive': torch.zeros((1, contrastive_L[idx], 2))},
             {'inpainting': torch.zeros((inpainting[idx]), dtype=torch.int64),
              'jigsaw': torch.zeros((jigsaw[idx]), dtype=torch.int64)})
            for idx in range(B)]

    x, target = collator(data)

    padding_mask_ref = torch.zeros((B, max(S), max(L)), dtype=torch.bool)
    padding_mask_ref[0, :, -1] = True
    padding_mask_ref[1, -1, :] = True
    padding_mask_contrastive_ref = torch.zeros((B, max(contrastive_S), max(contrastive_L)), dtype=torch.bool)
    padding_mask_contrastive_ref[1, -1, :] = True
    padding_mask_contrastive_ref[1, :, -3:] = True
    x_ref = {'msa': torch.zeros((B, max(S), max(L)), dtype=torch.int),
             'mask': torch.zeros((B, max(S), max(L)), dtype=torch.bool),
             'padding_mask': padding_mask_ref,
             'aux_features': torch.zeros((B, 1, max(L), 2)),
             'contrastive': torch.zeros((B, max(contrastive_S), max(contrastive_L)), dtype=torch.int),
             'padding_mask_contrastive': padding_mask_contrastive_ref,
             'aux_features_contrastive': torch.zeros((B, 1, max(contrastive_L), 2))}
    jigsaw_ref = torch.zeros((B, max(S)), dtype=torch.int64)
    jigsaw_ref[1, -1] = -1
    target_ref = {'inpainting': torch.zeros((sum(inpainting), ), dtype=torch.int64),
                  'jigsaw': jigsaw_ref}

    for key in x:
        if x[key].is_floating_point():
            testing.assert_close(x[key], x_ref[key])
        else:
            testing.assert_close(x[key], x_ref[key], atol=0, rtol=0)
    for key in target:
        if target[key].is_floating_point():
            testing.assert_close(target[key], target_ref[key])
        else:
            testing.assert_close(target[key], target_ref[key], atol=0, rtol=0)


if __name__ == '__main__':
    pass
