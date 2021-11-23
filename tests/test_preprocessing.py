from copy import deepcopy
import torch
import torch.testing as testing

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from selbstaufsicht.utils import rna2index
from selbstaufsicht.models.self_supervised.msa.modules import InpaintingHead
from selbstaufsicht.models.self_supervised.msa.transforms import RandomMSAMasking, ExplicitPositionalEncoding, RandomMSASubsampling, RandomMSAShuffling
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

    print('msa', x['msa'])
    print('mask', x['mask'])
    print("y", y['inpainting'])

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


def test_inpainting_head():
    num_classes = 4
    bs, S, L, D = 2, 3, 4, 8

    model = InpaintingHead(D, num_classes)
    latent = torch.rand((bs, S, L, D))
    mask = torch.full((bs, S, L), 0.5)
    mask = torch.bernoulli(mask).to(torch.bool)

    out = model(latent, {'mask': mask})
    out_ref = torch.tensor([[0.8108, 0.4546,  0.0308, -0.3818],
                            [0.7133, 0.7049, -0.0218, -0.3852],
                            [0.6086, 0.4439, -0.2997, -0.0855],
                            [0.5279, 0.2488, -0.1313, -0.0970],
                            [0.4448, 0.5628, -0.2541, -0.0900],
                            [0.4254, 0.5415, -0.1067, -0.0165],
                            [0.3169, 0.5469, -0.2521,  0.0763],
                            [0.5263, 0.7497, -0.0420, -0.3318],
                            [0.4400, 0.8195, -0.1103,  0.0479],
                            [0.5831, 0.7132, -0.2069, -0.2395],
                            [0.8939, 0.5034,  0.1073, -0.3953],
                            [0.5738, 0.5081, -0.0580, -0.1092],
                            [0.6805, 0.5888, -0.1333, -0.3499],
                            [0.7092, 0.4774,  0.0135, -0.2209],
                            [0.5564, 0.4920, -0.2069, -0.0589],
                            [0.7314, 0.1561, -0.0317, -0.2996]])

    testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-3)


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


def test_pad_collate_nd():
    # Test 2d
    shapes = [(4, 3), (3, 4)]
    batch = []
    for idx in range(len(shapes)):
        batch.append(torch.ones((shapes[idx])))
    out = _pad_collate_nd(batch)
    out_ref = torch.ones((len(shapes), 4, 4))
    out_ref[0, :, -1] = 0.
    out_ref[1, -1, :] = 0.
    testing.assert_close(out, out_ref, atol=0, rtol=0)

    # Test 3d
    shapes = [(6, 4, 2), (1, 3, 5)]
    batch = []
    for idx in range(len(shapes)):
        batch.append(torch.ones((shapes[idx])))
    out = _pad_collate_nd(batch)
    out_ref = torch.ones((len(shapes), 6, 4, 5))
    out_ref[0, :, :, -3:] = 0.
    out_ref[1, -5:, :, :] = 0.
    out_ref[1, :, -1:, :] = 0.
    testing.assert_close(out, out_ref, atol=0, rtol=0)


def test_msa_collator():
    collator = MSACollator()
    B = 2
    S = [5, 4]
    L = [6, 7]
    inpainting = [3, 2]

    # TODO: Test jigsaw and other tasks as well
    data = [({'msa': torch.zeros((S[idx], L[idx]), dtype=torch.int),
              'mask': torch.zeros((S[idx], L[idx]), dtype=torch.bool),
              'aux_features': torch.zeros((1, L[idx], 2))},
             {'inpainting': torch.zeros((inpainting[idx]), dtype=torch.int64)})
            for idx in range(B)]

    x, target = collator(data)

    x_ref = {'msa': torch.zeros((B, max(S), max(L)), dtype=torch.int),
             'mask': torch.zeros((B, max(S), max(L)), dtype=torch.bool),
             'aux_features': torch.zeros((B, 1, max(L), 2))}
    target_ref = {'inpainting': torch.zeros((sum(inpainting), ), dtype=torch.int64)}

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
