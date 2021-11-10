import os
import gzip
import urllib
import urllib.request


def get_family_ids(rfam_file):
    """
    extracts Rfam family IDs from a stockholm formatted multiple sequence alignment file
    """
    ids = []
    with gzip.open(rfam_file, 'rt', encoding='latin1') as f:
        for line in f:
            line = line.split()
            if len(line) > 0 and line[0] == '#=GF' and line[1] == 'AC':
                ids.append(line[2])
    return ids


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
