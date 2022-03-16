import torch
from torch.nn import CrossEntropyLoss
import torch.testing as testing
import pytest

from selbstaufsicht.modules import Accuracy, EmbeddedJigsawAccuracy, EmbeddedJigsawLoss, BinaryFocalNLLLoss, DiceNLLLoss, BinaryTopLPrecision, BinaryTopLF1Score, BinaryConfusionMatrix
from selbstaufsicht.utils import lehmer_encode, perm_metric, perm_gram_matrix, embed_finite_metric_space


def test_accuracy():
    preds = torch.tensor([[[ 1.0, -0.4,  0.2, -0.1],
                           [ 0.4,  1.0, -0.6, -0.2], 
                           [ 0.1, -0.2,  1.0,  0.3], 
                           [-0.7,  0.1, -0.3,  1.0],
                           [ 0.0,  0.0,  0.0,  0.0]], 
                          [[ 0.0,  0.0,  0.0,  0.0],
                           [ 0.3, -0.1,  0.2,  1.0],
                           [ 0.2, -0.9,  1.0, -0.3], 
                           [ 0.5,  1.0, -0.1,  0.6], 
                           [ 1.0,  0.4, -0.2,  0.8]]])
    target = torch.tensor([[0, 1, 2, 3, -1], [-1, 3, 2, 1, 0]], dtype=torch.int64)
    
    accuracy_metric = Accuracy()
    
    acc = accuracy_metric(preds, target)
    acc_ref = torch.tensor(1.0)
    
    testing.assert_close(acc, acc_ref, atol=1e-4, rtol=1e-3)
    
    # Error Case: invalid ignore_index
    accuracy_metric = Accuracy(ignore_index=2)
    with pytest.raises(AssertionError) as excinfo:
        acc = accuracy_metric(preds, target)
        
    assert str(excinfo.value) == "Parameter \'ignore_index\' must not be in range [0, num_classes)!"
    
    # Error Case: Different shapes
    preds = preds.repeat(2, 1, 1)
    accuracy_metric = Accuracy()
    with pytest.raises(AssertionError) as excinfo:
        acc = accuracy_metric(preds, target)
    
    assert str(excinfo.value) == "Shapes must match except for \'class_dim\'!"
    
    
def test_embedded_jigsaw_accuracy():
    euclid_emb = torch.Tensor([[  0.0,  0.0,  0.0,  0.0],
                               [  1.0,  1.0,  0.0,  0.0],
                               [  0.0,  0.0,  1.0,  1.0],
                               [  1.0,  1.0,  1.0,  1.0]])
    
    preds = torch.tensor([[[  1.0,  1.0,  1.0,  1.0],
                           [  0.0,  0.0,  0.0,  0.0]], 
                          [[  0.0,  0.0,  0.0,  0.0],
                           [  1.0,  1.0,  1.0,  1.0]],
                          [[  1.0,  1.0,  1.0,  1.0],
                           [  0.0,  0.0,  0.0,  0.0]],
                          [[ 42.0, 42.0, 42.0, 42.0],
                           [ 42.0, 42.0, 42.0, 42.0]]])
    target = torch.tensor([[[  1.0,  1.0,  0.0,  0.0],
                            [  0.0,  0.0,  1.0,  1.0]], 
                           [[  0.0,  0.0,  1.0,  1.0],
                            [  1.0,  1.0,  0.0,  0.0]],
                           [[  1.0,  1.0,  1.0,  1.0],
                            [  0.0,  0.0,  0.0,  0.0]],
                           [[ -1.0, -1.0, -1.0, -1.0],
                            [ -1.0, -1.0, -1.0, -1.0]]])
    
    accuracy_metric = EmbeddedJigsawAccuracy(euclid_emb)
    
    acc = accuracy_metric(preds, target)
    acc_ref = torch.tensor(2 / 6)
    
    testing.assert_close(acc, acc_ref, atol=1e-4, rtol=1e-3)
    
    # Error Case: Different shapes
    preds = preds.repeat(2, 1, 1)
    with pytest.raises(AssertionError) as excinfo:
        acc = accuracy_metric(preds, target)
    
    assert str(excinfo.value) == "Shapes must match!"
    
    
def test_embedded_jigsaw_loss():
    preds = torch.tensor([[[  1.0,  1.0,  1.0,  1.0],
                           [  0.0,  0.0,  0.0,  0.0]], 
                          [[  0.0,  0.0,  0.0,  0.0],
                           [  1.0,  1.0,  1.0,  1.0]],
                          [[  1.0,  1.0, 42.0, 42.0],
                           [  0.0,  0.0, 42.0, 42.0]],
                          [[ 42.0, 42.0, 42.0, 42.0],
                           [ 42.0, 42.0, 42.0, 42.0]]])
    target = torch.tensor([[[  1.0,  1.0,  0.0,  0.0],
                            [  0.0,  0.0,  1.0,  1.0]], 
                           [[  0.0,  0.0,  1.0,  1.0],
                            [  1.0,  1.0,  0.0,  0.0]],
                           [[  1.0,  0.0, -1.0, -1.0],
                            [  0.0,  1.0, -1.0, -1.0]],
                           [[ -1.0, -1.0, -1.0, -1.0],
                            [ -1.0, -1.0, -1.0, -1.0]]])
    
    loss_metric = EmbeddedJigsawLoss(reduction='mean')
    loss = loss_metric(preds, target)
    loss_ref = torch.tensor(10.0) / 24
    testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-3)
    
    loss_metric = EmbeddedJigsawLoss(reduction='sum')
    loss = loss_metric(preds, target)
    loss_ref = torch.tensor(10.0)
    testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-3)
    
    loss_metric = EmbeddedJigsawLoss(reduction='none')
    loss = loss_metric(preds, target)
    loss_ref = torch.tensor([[[ 0.0, 0.0, 1.0, 1.0],
                              [ 0.0, 0.0, 1.0, 1.0]], 
                             [[ 0.0, 0.0, 1.0, 1.0],
                              [ 0.0, 0.0, 1.0, 1.0]],
                             [[ 0.0, 1.0, 0.0, 0.0],
                              [ 0.0, 1.0, 0.0, 0.0]],
                             [[ 0.0, 0.0, 0.0, 0.0],
                              [ 0.0, 0.0, 0.0, 0.0]]])
    testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-3)
    
    # Error Case: Different shapes
    preds = preds.repeat(2, 1, 1)
    loss_metric = EmbeddedJigsawLoss()
    with pytest.raises(AssertionError) as excinfo:
        loss = loss_metric(preds, target)
    
    assert str(excinfo.value) == "Shapes must match!"
    

def test_perm_metric():
    perm1 = torch.tensor([0, 1, 2, 3])
    perm2 = perm1.clone()
    
    dist = perm_metric(perm1, perm2)
    assert dist == 0
    
    perm2 = torch.tensor([1, 0, 2, 3])
    dist = perm_metric(perm1, perm2)
    assert dist == 1
    
    perm2 = torch.tensor([1, 0, 3, 2])
    dist = perm_metric(perm1, perm2)
    assert dist == 2
    
    perm2 = torch.tensor([3, 0, 1, 2])
    dist = perm_metric(perm1, perm2)
    assert dist == 3


def test_perm_gram_matrix():
    perms = torch.vstack([lehmer_encode(i, 3) for i in range(6)])
    perms_ref = torch.tensor([[0, 1, 2],
                              [0, 2, 1],
                              [1, 0, 2],
                              [1, 2, 0],
                              [2, 0, 1],
                              [2, 1, 0]])
    testing.assert_close(perms, perms_ref, rtol=0, atol=0)
    
    d0 = perm_gram_matrix(perms)
    d0_ref = torch.tensor([[0, 1, 1, 2, 2, 1],
                            [1, 0, 2, 1, 1, 2],
                            [1, 2, 0, 1, 1, 2],
                            [2, 1, 1, 0, 2, 1],
                            [2, 1, 1, 2, 0, 1],
                            [1, 2, 2, 1, 1, 0]])
    testing.assert_close(d0, d0_ref, rtol=0, atol=0)


def test_embed_finite_metric_space():
    perms = torch.vstack([lehmer_encode(i, 2) for i in range(2)])
    perms_ref = torch.tensor([[0, 1],
                              [1, 0]])
    testing.assert_close(perms, perms_ref, rtol=0, atol=0)
    
    d0 = perm_gram_matrix(perms)
    d0_ref = torch.tensor([[0, 1],
                            [1, 0]])
    testing.assert_close(d0, d0_ref, rtol=0, atol=0)
    
    emb = embed_finite_metric_space(d0)
    emb_ref = torch.tensor([[0.],
                            [1.]])

    testing.assert_close(emb, emb_ref, atol=1e-4, rtol=1e-3)

    perms = torch.vstack([lehmer_encode(i, 4) for i in range(24)])
    d0 = perm_gram_matrix(perms)
    emb = embed_finite_metric_space(d0)
    assert torch.all(emb >= 0)
    

def test_binary_top_l_precision():
    top_l_precision_metric = BinaryTopLPrecision(diag_shift=1)
    preds = torch.Tensor([[[[0.1, 0.4, 0.3, 0.8], 
                            [0.4, 0.2, 0.9, 0.7],
                            [0.3, 0.9, 0.1, 0.2],
                            [0.8, 0.7, 0.2, 0.2]], 
                           [[0.9, 0.6, 0.7, 0.2], 
                            [0.6, 0.8, 0.1, 0.3],
                            [0.7, 0.1, 0.9, 0.8],
                            [0.2, 0.3, 0.8, 0.8]]],
                          [[[0.9, 0.6, 0.7, 0.2], 
                            [0.6, 0.8, 0.1, 0.3],
                            [0.7, 0.1, 0.9, 0.8],
                            [0.2, 0.3, 0.8, 0.8]], 
                           [[0.1, 0.4, 0.3, 0.8], 
                            [0.4, 0.2, 0.9, 0.7],
                            [0.3, 0.9, 0.1, 0.2],
                            [0.8, 0.7, 0.2, 0.2]]]])
    target = torch.Tensor([[[ 1, 0,  1,  1], 
                            [ 0, 1,  0,  1],
                            [ 1, 0,  1,  0],
                            [ 1, 1,  0,  1]], 
                           [[ 0, 1,  0, -1], 
                            [ 1, 0,  1,  0],
                            [ 0, 1,  0, -1],
                            [-1, 0, -1,  0]]])
    
    top_l_precision = top_l_precision_metric(preds, target)
    tp = top_l_precision_metric.tp
    fp = top_l_precision_metric.fp
    
    assert tp == 1
    assert fp == 1
    assert top_l_precision == 1 / 2
    
    top_l_precision_metric = BinaryTopLPrecision(diag_shift=1, treat_all_preds_positive=True)
    top_l_precision = top_l_precision_metric(preds, target)
    tp = top_l_precision_metric.tp
    fp = top_l_precision_metric.fp
    
    assert tp == 3
    assert fp == 2
    assert top_l_precision == 3 / 5
    

def test_binary_top_l_f1_score():
    top_l_f1_score_metric = BinaryTopLF1Score(diag_shift=1)
    preds = torch.Tensor([[[[0.1, 0.4, 0.3, 0.8], 
                            [0.4, 0.2, 0.9, 0.7],
                            [0.3, 0.9, 0.1, 0.2],
                            [0.8, 0.7, 0.2, 0.2]], 
                           [[0.9, 0.6, 0.7, 0.2], 
                            [0.6, 0.8, 0.1, 0.3],
                            [0.7, 0.1, 0.9, 0.8],
                            [0.2, 0.3, 0.8, 0.8]]],
                          [[[0.9, 0.6, 0.7, 0.2], 
                            [0.6, 0.8, 0.1, 0.3],
                            [0.7, 0.1, 0.9, 0.8],
                            [0.2, 0.3, 0.8, 0.8]], 
                           [[0.1, 0.4, 0.3, 0.8], 
                            [0.4, 0.2, 0.9, 0.7],
                            [0.3, 0.9, 0.1, 0.2],
                            [0.8, 0.7, 0.2, 0.2]]]])
    target = torch.Tensor([[[ 1, 0,  1,  1], 
                            [ 0, 1,  0,  1],
                            [ 1, 0,  1,  0],
                            [ 1, 1,  0,  1]], 
                           [[ 0, 1,  0, -1], 
                            [ 1, 0,  1,  0],
                            [ 0, 1,  0, -1],
                            [-1, 0, -1,  0]]])
    
    top_l_f1_score = top_l_f1_score_metric(preds, target)
    tp = top_l_f1_score_metric.tp
    fp = top_l_f1_score_metric.fp
    fn = top_l_f1_score_metric.fn
    
    assert tp == 1
    assert fp == 1
    assert fn == 2
    assert top_l_f1_score == 2 / 5
    
    
def test_binary_confusion_matrix():
    confmat_metric = BinaryConfusionMatrix()
    preds = torch.Tensor([[[[0.0, 0.0, 0.0], 
                            [1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0]], 
                           [[1.0, 1.0, 1.0], 
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0]]],
                          [[[1.0, 0.5, 0.5], 
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5]], 
                           [[0.0, 0.5, 0.5], 
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5]]]])
    target = torch.Tensor([[[ 1,  0,  0], 
                            [ 0,  0,  0],
                            [ 1,  1,  1]], 
                           [[ 1, -1, -1], 
                            [-1, -1, -1],
                            [-1, -1, -1]]])
    
    confmat = confmat_metric(preds, target)
    tp = confmat_metric.tp
    fp = confmat_metric.fp
    tn = confmat_metric.tn
    fn = confmat_metric.fn
    
    assert tp == 1
    assert fp == 2
    assert tn == 3
    assert fn == 4
    confmat_ref = torch.tensor([[1, 4], [2, 3]])
    testing.assert_close(confmat, confmat_ref, rtol=0, atol=0)
    
    confmat_metric = BinaryConfusionMatrix(reduce=False)
    confmat_metric(preds[:1], target[:1])
    confmat_metric(preds[1:], target[1:])
    confmat = confmat_metric.compute()
    
    confmat_ref = torch.tensor([[[1, 0], [3, 1]], [[2, 0], [3, 0]]])
    testing.assert_close(confmat, confmat_ref, rtol=0, atol=0)
    
    
def test_binary_focal_nllloss():
    preds = torch.tensor([[[  1e-8,  1e-8,   1.0,   1.0],
                           [   1.0,   1.0,  1e-8,  1e-8]], 
                          [[   1.0,   1.0,  1e-8,  1e-8],
                           [  1e-8,  1e-8,   1.0,   1.0]],
                          [[   0.1,   0.9,  42.0,  42.0],
                           [   0.9,   0.1,  42.0,  42.0]],
                          [[  42.0,  42.0,  42.0,  42.0],
                           [  42.0,  42.0,  42.0,  42.0]]])
    preds = torch.log(preds)
    target = torch.tensor([[  1,  1,  0,  0], 
                           [  0,  0,  1,  1],
                           [  0,  1, -1, -1],
                           [ -1, -1, -1, -1]])
    
    el = 0.5 * -torch.log(torch.tensor(0.1))
    
    loss_metric = BinaryFocalNLLLoss(gamma=0., reduction='mean')
    loss_metric_ce = CrossEntropyLoss(weight=torch.tensor([0.5, 0.5]), ignore_index=-1, reduction='mean')
    loss = loss_metric(preds, target)
    loss_ref = 2 * el / (10 * 0.5)
    loss_ref_ce = loss_metric_ce(preds, target)
    testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-3)
    testing.assert_close(loss, loss_ref_ce, atol=1e-4, rtol=1e-3)
    
    loss_metric = BinaryFocalNLLLoss(gamma=0., reduction='sum')
    loss_metric_ce = CrossEntropyLoss(weight=torch.tensor([0.5, 0.5]), ignore_index=-1, reduction='sum')
    loss = loss_metric(preds, target)
    loss_ref = 2 * el
    loss_ref_ce = loss_metric_ce(preds, target)
    testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-3)
    testing.assert_close(loss, loss_ref_ce, atol=1e-4, rtol=1e-3)
    
    loss_metric = BinaryFocalNLLLoss(gamma=0., reduction='none')
    loss_metric_ce = CrossEntropyLoss(weight=torch.tensor([0.5, 0.5]), ignore_index=-1, reduction='none')
    loss = loss_metric(preds, target)
    loss_ref = torch.tensor([[ 0., 0., 0., 0.], 
                             [ 0., 0., 0., 0.],
                             [ el, el, 0., 0.],
                             [ 0., 0., 0., 0.]])
    loss_ref_ce = loss_metric_ce(preds, target)
    testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-3)
    testing.assert_close(loss, loss_ref_ce, atol=1e-4, rtol=1e-3)
    
    weight = torch.tensor([0.2, 0.8])
    el1 = weight[0] * -torch.log(torch.tensor(0.1))
    el2 = weight[1] * -torch.log(torch.tensor(0.1))
    loss_metric = BinaryFocalNLLLoss(weight=weight, gamma=0., reduction='none')
    loss_metric_ce = CrossEntropyLoss(weight=weight, ignore_index=-1, reduction='none')
    loss = loss_metric(preds, target)
    loss_ref = torch.tensor([[  0.,  0., 0., 0.], 
                             [  0.,  0., 0., 0.],
                             [ el1, el2, 0., 0.],
                             [  0.,  0., 0., 0.]])
    loss_ref_ce = loss_metric_ce(preds, target)
    testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-3)
    testing.assert_close(loss, loss_ref_ce, atol=1e-4, rtol=1e-3)
    
    el *= 0.9 ** 42
    loss_metric = BinaryFocalNLLLoss(gamma=42., reduction='none')
    loss = loss_metric(preds, target)
    loss_ref = torch.tensor([[ 0., 0., 0., 0.], 
                             [ 0., 0., 0., 0.],
                             [ el, el, 0., 0.],
                             [ 0., 0., 0., 0.]])
    testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-3)


def test_dice_nllloss():
    preds = torch.tensor([[[[1e-8,  1.0], 
                            [ 1.0, 1e-8]], 
                           [[ 1.0, 1e-8], 
                            [1e-8,  1.0]]], 
                          [[[ 0.1,  0.9], 
                            [42.0, 42.0]], 
                           [[ 0.9,  0.1], 
                            [42.0, 42.0]]]])
    preds = torch.log(preds)
    target = torch.Tensor([[[ 1,  0], 
                            [ 0,  1]], 
                           [[ 0,  1], 
                            [-1, -1]]])
    
    loss_metric = DiceNLLLoss(reduction='mean')
    loss = loss_metric(preds, target)
    loss_ref = torch.tensor(0.5875)
    testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-3)
    
    loss_metric = DiceNLLLoss(reduction='sum')
    loss = loss_metric(preds, target)
    loss_ref = torch.tensor(1.175)
    testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-3)
    
    loss_metric = DiceNLLLoss(reduction='none')
    loss = loss_metric(preds, target)
    loss_ref = torch.tensor([0.25, 0.925])
    testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-3)
