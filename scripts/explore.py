import os
from matplotlib import pyplot as plt

from selbstaufsicht import datasets


root = os.environ['DATA_PATH'] + 'Xfam'

ds = datasets.Xfam(root, download=True, mode='seed', version='14.6')

lnseqs = list()
lnbases = list()
lnbases_total = list()
nalignments = len(ds)

for a in ds:
    nseqs = len(a)
    nbases = a.get_alignment_length()

    lnseqs.append(nseqs)
    lnbases.append(nbases)
    lnbases_total.append(nseqs * nbases)


print('number of alignments/samples: ', nalignments)
print('number of sequences total: ', sum(lnseqs))
print('number of bases total: ', sum(lnbases_total))
print('min sequences per alignment: ', min(lnseqs))
print('max sequences per alignment: ', max(lnseqs))
print('max len sequence in alignment: ', max(lnbases))
print('min len sequence in alignment: ', min(lnbases))

plt.hist(lnseqs, bins=100)
plt.xlabel("sequences")
plt.ylabel("count")
# plt.savefig("rfam_sequences_dist.pdf")
# plt.show()

# plt.hist(lnbases, bins=100)
plt.hist(lnbases, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400])
plt.xlabel("bases")
plt.ylabel("count")
# plt.savefig("rfam_bases_dist.pdf")
# plt.show()

plt.hist(lnbases_total, bins=100)
plt.xlabel("bases")
plt.ylabel("count")
# plt.savefig("rfam_bases_total_dist.pdf")
# plt.show()
