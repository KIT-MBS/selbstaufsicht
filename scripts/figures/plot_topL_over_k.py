import numpy as np
import matplotlib.pyplot as plt


def plot_individual_msa_abs_data(pdb_id):
    x = np.load('%s_x_abs.npy' % pdb_id)
    y = np.load('%s_y_abs.npy' % pdb_id)
    
    plt.plot(x, y, 'b-')
    plt.xlabel("k")
    plt.ylabel("Top-k-Precision")
    plt.xscale('log')
    plt.title("Top-k-Precision for absolute k (PDB: %s)" % pdb_id)
    plt.show()
    

def plot_all_msa_rel_data():
    x = np.load('x_rel.npy')
    y = np.load('y_rel.npy')
    std = np.load('std_rel.npy')
    
    plt.plot(x, y, 'r-')
    plt.fill_between(x, y - std, y + std, color='r', alpha=0.2)
    plt.xlabel("k")
    plt.ylabel("Top-(k*L)-Precision")
    plt.xscale('log')
    plt.title("Top-(k*L)-Precision for relative k (all MSAs)")
    plt.show()


def main():
    pdb_ids = ['1c2x', '1l9a', '1p6v', '1s9s', '1z2j', '2lc8', '2mf0', '2nbx', '3ndb', '4gma', '5ddp', '5di2', '5zal', '6hag', '6ol3', '6qn3', '6ues']
    
    plot_all_msa_rel_data()
    for pdb_id in pdb_ids:
        plot_individual_msa_abs_data(pdb_id)


if __name__ == '__main__':
    main()