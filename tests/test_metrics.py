import torch
import torch.testing as testing
import pytest

from selbstaufsicht.modules import Accuracy, EmbeddedJigsawLoss
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