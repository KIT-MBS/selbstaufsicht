import torch
import torch.testing as testing
import pytest

from selbstaufsicht.modules.metrics import Accuracy


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