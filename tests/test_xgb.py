import numpy as np
import numpy.testing as testing
import xgboost as xgb

from selbstaufsicht.models.xgb import xgb_contact


def test_sigmoid():
    x = np.array([-np.log(42), 0, np.log(42)])
    
    y = xgb_contact.sigmoid(x)
    y_ref = np.array([1/43, 0.5, 1 - 1/43])
    
    testing.assert_allclose(y, y_ref)


def test_xgb_topkLPrec():
    preds = np.array([np.log(0.7/0.3), 
                      np.log(0.2/0.8), 
                      np.log(0.3/0.7), 
                      np.log(0.3/0.7), 
                      np.log(0.7/0.3)])
    dmat = xgb.DMatrix(np.array([[]]), label=np.array([1, 1, 1, 0, 0]))
    msa_mapping = np.array([0, 0, 0, 1, 1])
    L_mapping = np.array([4, 4])
    
    top_l_prec = xgb_contact.xgb_topkLPrec(preds, dmat, msa_mapping, L_mapping)
    assert top_l_prec == 1 / 2
    
    top_l_prec = xgb_contact.xgb_topkLPrec(preds, dmat, msa_mapping, L_mapping, treat_all_preds_positive=True)
    assert top_l_prec == 3 / 5
    
    
def test_xgb_precision():
    preds = np.array([np.log(0.7/0.3), 
                      np.log(0.2/0.8), 
                      np.log(0.3/0.7), 
                      np.log(0.3/0.7), 
                      np.log(0.7/0.3)])
    dmat = xgb.DMatrix(np.array([[]]), label=np.array([1, 1, 1, 0, 0]))
    msa_mapping = np.array([0, 0, 0, 1, 1])
    
    top_l_prec = xgb_contact.xgb_precision(preds, dmat, msa_mapping)
    assert top_l_prec == 1 / 2
    

def test_xgb_recall():
    preds = np.array([np.log(0.7/0.3), 
                      np.log(0.2/0.8), 
                      np.log(0.3/0.7), 
                      np.log(0.3/0.7), 
                      np.log(0.7/0.3)])
    dmat = xgb.DMatrix(np.array([[]]), label=np.array([1, 1, 1, 0, 0]))
    msa_mapping = np.array([0, 0, 0, 1, 1])
    
    recall = xgb_contact.xgb_recall(preds, dmat, msa_mapping)
    assert recall == 1 / 3
    

def test_xgb_F1Score():
    preds = np.array([np.log(0.7/0.3), 
                      np.log(0.2/0.8), 
                      np.log(0.3/0.7), 
                      np.log(0.3/0.7), 
                      np.log(0.7/0.3)])
    dmat = xgb.DMatrix(np.array([[]]), label=np.array([1, 1, 1, 0, 0]))
    msa_mapping = np.array([0, 0, 0, 1, 1])
    
    f1_score = xgb_contact.xgb_F1Score(preds, dmat, msa_mapping)
    assert f1_score == 2 / 5
    

def test_xgb_Matthews():
    preds = np.array([np.log(0.7/0.3), 
                      np.log(0.2/0.8), 
                      np.log(0.3/0.7), 
                      np.log(0.3/0.7), 
                      np.log(0.7/0.3)])
    dmat = xgb.DMatrix(np.array([[]]), label=np.array([1, 1, 1, 0, 0]))
    msa_mapping = np.array([0, 0, 0, 1, 1])
    
    mcc = xgb_contact.xgb_Matthews(preds, dmat, msa_mapping)
    assert mcc == -1 / 6