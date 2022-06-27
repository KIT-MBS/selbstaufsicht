import os
import gzip
import urllib
import urllib.request
import subprocess
from Bio import AlignIO
from Bio import SeqIO
from selbstaufsicht.datasets._utils import get_family_ids

root = os.environ['DATA_PATH'] + '/Xfam'
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


rfam_ids = get_family_ids(path + filename)

with gzip.open(path + filename, 'rt', encoding='latin1') as f:
    for i, a in enumerate(AlignIO.parse(f, 'stockholm')):
        if len(a) < 100:
            fasta_filepath = root + '/Rfam/14.6/fasta_files/' + rfam_ids[i] + '.fa.gz'
            fasta_url = fasta_base_url + rfam_ids[i] + '.fa.gz'

            if not os.path.isfile(fasta_filepath):
                print(fasta_filepath)
                _download(fasta_url, fasta_filepath)

            with gzip.open(fasta_filepath, 'rt', encoding='latin1') as ff:
                seq_records = [sr for sr in SeqIO.parse(ff, 'fasta')]
                print(rfam_ids[i], len(a), len(seq_records))
                if len(a) < len(seq_records):
                    full_file_path = root + f'/Rfam/14.6/full/train/{rfam_ids[i]}.sto'
                    cm_path = root + f'/Rfam/14.6/cms/{rfam_ids[i]}.cm'
                    if not os.path.isfile(full_file_path):
                        print("AAAAAAA")
                        subprocess.call(['cmsearch', '-A', full_file_path, cm_path, fasta_filepath])
