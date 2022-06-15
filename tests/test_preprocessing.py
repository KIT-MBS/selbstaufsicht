from copy import deepcopy
import torch
from torch.distributions import Categorical
import torch.testing as testing

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from selbstaufsicht.transforms import SelfSupervisedCompose
from selbstaufsicht.utils import rna2index, nonstatic_mask_tokens
from selbstaufsicht.models.self_supervised.msa.transforms import MSATokenize, _get_replace_mask, RandomMSAMasking, ExplicitPositionalEncoding, MSACropping, MSASubsampling, RandomMSAShuffling
from selbstaufsicht.models.self_supervised.msa.transforms import DistanceFromChain, ContactFromDistance
from selbstaufsicht.models.self_supervised.msa.utils import MSACollator, _pad_collate_nd


def test_msa_tokenize(tokenized_sample):
    tokenized_msa_ref = torch.tensor([[17, 3, 5, 4, 5, 5, 4, 3],
                                      [17, 3, 3, 4, 1, 5, 4, 3],
                                      [17, 5, 5, 4, 3, 5, 4, 1],
                                      [17, 4, 5, 4, 5, 5, 4, 5]])
    testing.assert_close(tokenized_sample[0]['msa'], tokenized_msa_ref, rtol=0, atol=0)


def test_get_replace_mask():
    mask = torch.tensor([[0, 1, 1, 1],
                         [1, 0, 1, 1],
                         [1, 1, 0, 1],
                         [1, 1, 1, 0]], dtype=torch.bool)
    masking_type_sampling = torch.tensor([[0, 1, 2, 2],
                                          [0, 1, 2, 2],
                                          [0, 1, 2, 2],
                                          [0, 1, 2, 2]])
    static_mask_token = 42
    nonstatic_mask_tokens = [11, 22, 33]

    replace_mask = _get_replace_mask(mask, masking_type_sampling, static_mask_token, nonstatic_mask_tokens)
    replace_mask_ref = torch.tensor([42, 11, 33, 22, 22, 42, 11, 42, 33])
    mask_ref = torch.tensor([[0, 1, 1, 1],
                             [0, 0, 1, 1],
                             [0, 1, 0, 1],
                             [0, 1, 1, 0]], dtype=torch.bool)

    assert mask[mask == 1].numel() == replace_mask.numel()
    testing.assert_close(replace_mask, replace_mask_ref, atol=0, rtol=0)
    testing.assert_close(mask, mask_ref, atol=0, rtol=0)

    N = 1_000_000
    masking_type_p = torch.Tensor([0.1, 0.3, 0.6])
    mask = torch.ones((N,), dtype=torch.bool)
    masking_type_distribution = Categorical(masking_type_p)
    masking_type_sampling = masking_type_distribution.sample(mask.size())

    replace_mask = _get_replace_mask(mask, masking_type_sampling, static_mask_token, nonstatic_mask_tokens)
    ratio_unchanged = mask[mask == 0].numel() / N
    ratio_static = replace_mask[replace_mask == static_mask_token].numel() / N
    ratio_nonstatic_1 = replace_mask[replace_mask == nonstatic_mask_tokens[0]].numel() / N
    ratio_nonstatic_2 = replace_mask[replace_mask == nonstatic_mask_tokens[1]].numel() / N
    ratio_nonstatic_3 = replace_mask[replace_mask == nonstatic_mask_tokens[2]].numel() / N

    assert mask[mask == 1].numel() == replace_mask.numel()
    testing.assert_close(ratio_unchanged, masking_type_p[0].item(), atol=1e-3, rtol=1e-2)
    testing.assert_close(ratio_static, masking_type_p[1].item(), atol=1e-3, rtol=1e-2)
    testing.assert_close(ratio_nonstatic_1, masking_type_p[2].item() / 3, atol=1e-3, rtol=1e-2)
    testing.assert_close(ratio_nonstatic_2, masking_type_p[2].item() / 3, atol=1e-3, rtol=1e-2)
    testing.assert_close(ratio_nonstatic_3, masking_type_p[2].item() / 3, atol=1e-3, rtol=1e-2)


def test_msa_mask_token_static(tokenized_sample):
    masking = RandomMSAMasking(p=.5, p_static=1., p_nonstatic=0., p_unchanged=0., mode='token',
                               static_mask_token=rna2index['MASK_TOKEN'], nonstatic_mask_tokens=nonstatic_mask_tokens)
    positional = ExplicitPositionalEncoding()

    x, y = masking(*tokenized_sample)
    x, y = positional(x, y)

    x_ref = {'aux_features': torch.tensor([[1,
                                            2,
                                            3,
                                            4,
                                            5,
                                            6,
                                            7,
                                            8]]),
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


def test_msa_mask_token_nonstatic(tokenized_sample):
    masking = RandomMSAMasking(p=.5, p_static=0., p_nonstatic=1., p_unchanged=0., mode='token',
                               static_mask_token=rna2index['MASK_TOKEN'], nonstatic_mask_tokens=nonstatic_mask_tokens)
    positional = ExplicitPositionalEncoding()

    x, y = masking(*tokenized_sample)
    x, y = positional(x, y)

    x_ref = {'aux_features': torch.tensor([[1,
                                            2,
                                            3,
                                            4,
                                            5,
                                            6,
                                            7,
                                            8]]),
             'msa': torch.tensor([[17, 3, 5, 4, 0, 5, 5, 3],
                                  [17, 0, 3, 4, 1, 5, 4, 3],
                                  [17, 5, 4, 4, 2, 3, 2, 1],
                                  [17, 5, 3, 2, 5, 3, 4, 4]]),
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


def test_msa_mask_column_static(tokenized_sample):
    masking = RandomMSAMasking(p=.5, p_static=1., p_nonstatic=0., p_unchanged=0., mode='column',
                               static_mask_token=rna2index['MASK_TOKEN'], nonstatic_mask_tokens=nonstatic_mask_tokens)
    positional = ExplicitPositionalEncoding()

    x, y = masking(*tokenized_sample)
    x, y = positional(x, y)

    x_ref = {'aux_features': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
             'msa': torch.tensor([[17, 3, 5, 19, 5, 19, 4, 19],
                                  [17, 3, 3, 19, 1, 19, 4, 19],
                                  [17, 5, 5, 19, 3, 19, 4, 19],
                                  [17, 4, 5, 19, 5, 19, 4, 19]]),
             'mask': torch.tensor([[False, False, False, True, False, True, False, True],
                                   [False, False, False, True, False, True, False, True],
                                   [False, False, False, True, False, True, False, True],
                                   [False, False, False, True, False, True, False, True]])}

    y_ref = {'inpainting': torch.tensor([4, 5, 3, 4, 5, 3, 4, 5, 1, 4, 5, 5])}

    testing.assert_close(
        x['aux_features'], x_ref['aux_features'], atol=1e-4, rtol=1e-3)
    testing.assert_close(x['msa'], x_ref['msa'], rtol=0, atol=0)
    testing.assert_close(x['mask'], x_ref['mask'], rtol=0, atol=0)
    testing.assert_close(y['inpainting'], y_ref['inpainting'], rtol=0, atol=0)


def test_msa_mask_column_nonstatic(tokenized_sample):
    masking = RandomMSAMasking(p=.5, p_static=0., p_nonstatic=1., p_unchanged=0., mode='column',
                               static_mask_token=rna2index['MASK_TOKEN'], nonstatic_mask_tokens=nonstatic_mask_tokens)
    positional = ExplicitPositionalEncoding()

    x, y = masking(*tokenized_sample)
    x, y = positional(x, y)

    x_ref = {'aux_features': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
             'msa': torch.tensor([[17, 3, 5, 0, 5, 0, 4, 4],
                                  [17, 3, 3, 4, 1, 0, 4, 4],
                                  [17, 5, 5, 2, 3, 4, 4, 5],
                                  [17, 4, 5, 0, 5, 5, 4, 5]]),
             'mask': torch.tensor([[False, False, False, True, False, True, False, True],
                                   [False, False, False, True, False, True, False, True],
                                   [False, False, False, True, False, True, False, True],
                                   [False, False, False, True, False, True, False, True]])}

    y_ref = {'inpainting': torch.tensor([4, 5, 3, 4, 5, 3, 4, 5, 1, 4, 5, 5])}

    testing.assert_close(
        x['aux_features'], x_ref['aux_features'], atol=1e-4, rtol=1e-3)
    testing.assert_close(x['msa'], x_ref['msa'], rtol=0, atol=0)
    testing.assert_close(x['mask'], x_ref['mask'], rtol=0, atol=0)
    testing.assert_close(y['inpainting'], y_ref['inpainting'], rtol=0, atol=0)


def test_msa_mask_block_static(tokenized_sample):
    masking = RandomMSAMasking(p=.5, p_static=1., p_nonstatic=0., p_unchanged=0., mode='block',
                               static_mask_token=rna2index['MASK_TOKEN'], nonstatic_mask_tokens=nonstatic_mask_tokens)
    positional = ExplicitPositionalEncoding()

    x, y = masking(*tokenized_sample)
    x, y = positional(x, y)

    x_ref = {'aux_features': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
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


def test_msa_mask_block_nonstatic(tokenized_sample):
    masking = RandomMSAMasking(p=.5, p_static=0., p_nonstatic=1., p_unchanged=0., mode='block',
                               static_mask_token=rna2index['MASK_TOKEN'], nonstatic_mask_tokens=nonstatic_mask_tokens)
    positional = ExplicitPositionalEncoding()

    x, y = masking(*tokenized_sample)
    x, y = positional(x, y)

    x_ref = {'aux_features': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
             'msa': torch.tensor([[17, 3, 5, 5, 4, 3, 4, 3],
                                  [17, 3, 3, 5, 5, 2, 4, 3],
                                  [17, 5, 5, 0, 0, 4, 4, 1],
                                  [17, 4, 5, 4, 0, 4, 4, 5]]),
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


def test_jigsaw(tokenized_sample):
    permutations = torch.tensor([[0, 1],
                                 [1, 0]], dtype=torch.int64)
    label = torch.tensor([0, 1, 0, 1])
    shuffling = RandomMSAShuffling(permutations=permutations)
    x, y = shuffling(deepcopy(tokenized_sample)[0], {'jigsaw': label})

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
    x, y = shuffling(deepcopy(tokenized_sample)[0], {'jigsaw': label})

    x_ref = torch.tensor([[17, 5, 4, 3, 5, 4, 5, 3],
                          [17, 3, 3, 5, 4, 4, 1, 3],
                          [17, 4, 3, 5, 5, 5, 4, 1],
                          [17, 4, 5, 4, 5, 5, 4, 5]])

    testing.assert_close(x['msa'], x_ref, rtol=0, atol=0)
    testing.assert_close(y['jigsaw'], label, rtol=0, atol=0)


def test_jigsaw_delimiter(tokenized_sample):
    delimiter_token = rna2index['DELIMITER_TOKEN']
    permutations = torch.tensor([[0, 1],
                                 [1, 0]], dtype=torch.int64)
    label = torch.tensor([0, 1, 0, 1])
    shuffling = RandomMSAShuffling(permutations=permutations, delimiter_token=delimiter_token)
    x, y = shuffling(deepcopy(tokenized_sample)[0], {'jigsaw': label})

    x_ref = torch.tensor([[17, 18, 3, 5, 4, 18, 5, 5, 4, 18, 3],
                          [17, 18, 1, 5, 4, 18, 3, 3, 4, 18, 3],
                          [17, 18, 5, 5, 4, 18, 3, 5, 4, 18, 1],
                          [17, 18, 5, 5, 4, 18, 4, 5, 4, 18, 5]])

    testing.assert_close(x['msa'], x_ref, rtol=0, atol=0)
    testing.assert_close(y['jigsaw'], label, rtol=0, atol=0)

    permutations = torch.tensor([[0, 1, 2],
                                 [1, 0, 2],
                                 [0, 2, 1],
                                 [2, 0, 1]], dtype=torch.int64)
    label = torch.tensor([3, 2, 1, 0])
    shuffling = RandomMSAShuffling(permutations=permutations, delimiter_token=delimiter_token)
    x, y = shuffling(deepcopy(tokenized_sample)[0], {'jigsaw': label})

    x_ref = torch.tensor([[17, 18, 5, 4, 18, 3, 5, 18, 4, 5, 18, 3],
                          [17, 18, 3, 3, 18, 5, 4, 18, 4, 1, 18, 3],
                          [17, 18, 4, 3, 18, 5, 5, 18, 5, 4, 18, 1],
                          [17, 18, 4, 5, 18, 4, 5, 18, 5, 4, 18, 5]])

    testing.assert_close(x['msa'], x_ref, rtol=0, atol=0)
    testing.assert_close(y['jigsaw'], label, rtol=0, atol=0)


def test_subsampling_uniform(msa_sample):
    sampler = MSASubsampling(3, False, 'uniform')
    sampled = sampler(*msa_sample)[0]['msa']

    sampled_ref = MultipleSeqAlignment(
        [
            SeqRecord(Seq("ACUCCUA"), id='seq1'),
            SeqRecord(Seq("AAU.CUA"), id='seq2'),
            SeqRecord(Seq("UCUCCUC"), id='seq4'),
        ])

    for idx in range(len(sampled)):
        assert sampled[idx].seq == sampled_ref[idx].seq


def test_subsampling_maximum_diversity(msa_sample):
    sampler = MSASubsampling(3, False, 'diversity')
    x, y = msa_sample
    x['indices'] = torch.tensor([0, 2, 3], dtype=torch.int64)
    sampled = sampler(x, y)[0]['msa']

    sampled_ref = MultipleSeqAlignment(
        [
            SeqRecord(Seq("ACUCCUA"), id='seq1'),
            SeqRecord(Seq("CCUACU."), id='seq3'),
            SeqRecord(Seq("UCUCCUC"), id='seq4'),
        ])

    for idx in range(len(sampled)):
        assert sampled[idx].seq == sampled_ref[idx].seq


def test_subsampling_fixed(msa_sample):
    sampler = MSASubsampling(2, True, 'fixed')
    sampled = sampler(*msa_sample)[0]['msa']
    sampled_contrastive = sampler(*msa_sample)[0]['contrastive']

    sampled_ref = MultipleSeqAlignment(
        [
            SeqRecord(Seq("ACUCCUA"), id='seq1'),
            SeqRecord(Seq("AAU.CUA"), id='seq2'),
        ])

    sampled_contrastive_ref = MultipleSeqAlignment(
        [
            SeqRecord(Seq("CCUACU."), id='seq3'),
            SeqRecord(Seq("UCUCCUC"), id='seq4'),
        ])

    for idx in range(len(sampled)):
        assert sampled[idx].seq == sampled_ref[idx].seq

    for idx in range(len(sampled_contrastive)):
        assert sampled_contrastive[idx].seq == sampled_contrastive_ref[idx].seq


def test_cropping_random_dependent(msa_sample):
    cropper = MSACropping(3, False, 'random-dependent')
    cropped = cropper(*msa_sample)

    cropped_ref = MultipleSeqAlignment(
        [
            SeqRecord(Seq("UCC"), id='seq1'),
            SeqRecord(Seq("U.C"), id='seq2'),
            SeqRecord(Seq("UAC"), id='seq3'),
            SeqRecord(Seq("UCC"), id='seq4'),
        ]
    )

    for idx in range(len(cropped)):
        assert cropped[0]['msa'][idx].seq == cropped_ref[idx].seq


def test_cropping_random_independent(msa_sample):
    cropper = MSACropping(3, False, 'random-independent')
    cropped = cropper(*msa_sample)

    cropped_ref = MultipleSeqAlignment(
        [
            SeqRecord(Seq("UCC"), id='seq1'),
            SeqRecord(Seq(".CU"), id='seq2'),
            SeqRecord(Seq("CCU"), id='seq3'),
            SeqRecord(Seq("UCC"), id='seq4'),
        ]
    )

    for idx in range(len(cropped)):
        assert cropped[0]['msa'][idx].seq == cropped_ref[idx].seq


def test_cropping_fixed(msa_sample):
    cropper = MSACropping(3, False, 'fixed')
    cropped = cropper(*msa_sample)

    cropped_ref = MultipleSeqAlignment(
        [
            SeqRecord(Seq("ACU"), id='seq1'),
            SeqRecord(Seq("AAU"), id='seq2'),
            SeqRecord(Seq("CCU"), id='seq3'),
            SeqRecord(Seq("UCU"), id='seq4'),
        ]
    )

    for idx in range(len(cropped)):
        assert cropped[0]['msa'][idx].seq == cropped_ref[idx].seq


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


def test_compose(msa_sample):
    sampler = MSASubsampling(3, False, 'uniform')
    tokenize = MSATokenize(rna2index)

    transforms = [sampler, tokenize]
    transform_composition = SelfSupervisedCompose(transforms)
    sample, target = transform_composition(*msa_sample)

    msa_ref = torch.tensor([[17, 3, 5, 4, 5, 5, 4, 3],   # seq1
                            [17, 3, 3, 4, 1, 5, 4, 3],   # seq2
                            [17, 4, 5, 4, 5, 5, 4, 5]])  # seq4
    print(sample['msa'])

    testing.assert_close(sample['msa'], msa_ref, rtol=0, atol=0)


def test_distance_from_chain(bio_structure):
    dfc = DistanceFromChain(3)
    x = None
    y = {'structure': bio_structure}
    x, y = dfc(x, y)

    distances_ref = torch.tensor([
        [0., 1., 0., torch.inf, 4.],
        [1., 0., 1., torch.inf, 3.],
        [0., 1., 0., torch.inf, 4.],
        [torch.inf, torch.inf, torch.inf, torch.inf, torch.inf],
        [4., 3., 4., torch.inf, 0.],
    ], device=y['distances'].device)

    testing.assert_close(y['distances'], distances_ref)

    cfd = ContactFromDistance(1.5)
    x, y = cfd(x, y)
    contacts_ref = torch.tensor([
        [1, 1, 1, -1, 0],
        [1, 1, 1, -1, 0],
        [1, 1, 1, -1, 0],
        [-1, -1, -1, -1, -1],
        [0, 0, 0, -1, 1],
    ], device=y['distances'].device, dtype=torch.long)

    testing.assert_close(y['contact'], contacts_ref)


if __name__ == '__main__':
    pass
