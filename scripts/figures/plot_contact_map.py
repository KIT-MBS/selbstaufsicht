import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# other=0=black, TP=1=yellow, FP=2=red, TN=3=black, FN=4=blue
TP = [1., 1., 0.]
FP = [1., 0., 0.]
TN = [0., 0., 0.]
FN = [0., 0., 1.]
c_map = [[0., 0., 0.], TP, FP, TN, FN]


def eval_preds_target(preds, target):
    out = np.zeros_like(target).astype(int)
    
    tp_mask = np.logical_and(preds == 1, target == 1)
    fp_mask = np.logical_and(preds == 1, target == 0)
    tn_mask = np.logical_and(preds == 0, target == 0)
    fn_mask = np.logical_and(preds == 0, target == 1)
    
    out[tp_mask] = 1
    out[fp_mask] = 2
    out[tn_mask] = 3
    out[fn_mask] = 4
    
    return out


def plot(preds_lower, target_lower, preds_upper, target_upper, title=""):
    lower = np.tril(eval_preds_target(preds_lower, target_lower), -1)
    upper = np.triu(eval_preds_target(preds_upper, target_upper), 1)
    contact_map = lower + upper 
    cm = ListedColormap(c_map)
    plt.imshow(contact_map, origin='lower', interpolation='nearest', cmap=cm)
    
    tp_patch = Patch(color=TP, label="TP")
    fp_patch = Patch(color=FP, label="FP")
    tn_patch = Patch(color=TN, label="TN")
    fn_patch = Patch(color=FN, label="FN")
    plt.legend(handles=[tp_patch, fp_patch, tn_patch, fn_patch], loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.title(title)
    plt.show()


def main():
    preds_3d = np.load('3ndb_preds_3D.npy')
    preds_frozen = np.load('3ndb_preds_frozen.npy')
    preds = np.load('3ndb_preds.npy')
    target_3d = np.load('3ndb_target_3D.npy')
    target = np.load('3ndb_target.npy')
    
    plot(preds.T, target.T, preds_3d, target_3d, "Contact Map (PDB: 3ndb) - unfrozen_all vs. unfrozen_3D")
    plot(preds.T, target.T, preds_frozen, target, "Contact Map (PDB: 3ndb) - unfrozen_all vs. frozen_all")


if __name__ == '__main__':
    main()