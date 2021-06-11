import os

from selbstaufsicht import datasets


root = os.environ['DATA_PATH']

ds = datasets.Xfam(root, download=True)

lnseqs = list()
lnbases = list()
lnbases_total = list()
nalignments = len(ds)

for a in ds:
    nseqs = len(a)
    nbases = a.get_alignment_length()

    lnseqs.append(nseqs)
    lnbases.append(nbases)
    lnbases_total.append(nseqs*nbases)


from matplotlib import pyplot as plt

print('number of alignments/samples: ', nalignments)
print('number of sequences total: ', sum(lnseqs))
print('number of bases total: ', sum(lnbases_total))
print('max sequences per alignment: ', max(lnseqs))
print('max len sequence in alignment: ', max(lnbases))

plt.hist(lnseqs, bins=100)
plt.xlabel("sequences")
plt.ylabel("count")
# plt.savefig("rfam_sequences_dist.pdf")
plt.show()

plt.hist(lnbases, bins=100)
plt.xlabel("bases")
plt.ylabel("count")
# plt.savefig("rfam_bases_dist.pdf")
plt.show()

plt.hist(lnbases_total, bins=100)
plt.xlabel("bases")
plt.ylabel("count")
# plt.savefig("rfam_bases_total_dist.pdf")
plt.show()
