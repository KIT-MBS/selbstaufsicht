import os
import gzip
import urllib
import urllib.request
from Bio import AlignIO
from Bio import SeqIO

root = os.environ['DATA_PATH'] + 'Xfam'
path = root + '/Rfam/14.6/seed/train/'
filename = 'Rfam.gz'

fasta_base_url = 'ftp://ftp.ebi.ac.uk/pub/databases/Rfam/14.6/fasta_files/'


def _download(url, path):
    if os.path.isfile(path):
        print(f"Found existing dataset file; {path}.")
        return
    print(f"downloading from {url}")
    prefix, filename = os.path.split(path)
    os.makedirs(prefix, exist_ok=True)
    chunk_size = 1024
    with urllib.request.urlopen(urllib.request.Request(url)) as response:
        with open(path, 'wb') as f:
            for chunk in iter(lambda: response.read(chunk_size), ''):
                if not chunk:
                    break
                f.write(chunk)


def get_family_ids(rfam_file):
    ids = []
    with gzip.open(rfam_file, 'rt', encoding='latin1') as f:
        for line in f:
            line = line.split()
            if len(line) > 0 and line[0] == '#=GF' and line[1] == 'AC':
                ids.append(line[2])
    return ids


rfam_ids = get_family_ids(path + filename)

with gzip.open(path + filename, 'rt', encoding='latin1') as f:
    for i, a in enumerate(AlignIO.parse(f, 'stockholm')):
        if len(a) < 100:
            fasta_filepath = root + 'Rfam/14.6/fasta_files/' + rfam_ids[i] + '.fa.gz'
            fasta_url = fasta_base_url + rfam_ids[i] + '.fa.gz'

            if not os.path.isfile(fasta_filepath):
                print(fasta_filepath)
                _download(fasta_url, fasta_filepath)

            with gzip.open(fasta_filepath, 'rt', encoding='latin1') as ff:
                seq_records = [sr for sr in SeqIO.parse(ff, 'fasta')]
                print(rfam_ids[i], len(a), len(seq_records))
                if len(a) == len(seq_records):
                    print("match =(")
                elif len(a) < len(seq_records):
                    print("hurraaaaaaaaaaaaaaaay")
                else:
                    print('wtf?')
