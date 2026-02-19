# Data_encoded.py
# Encode raw nucleosome/linker FASTA sequences into pickle files.
#
# Usage:
#   python Data_encoded.py \
#       -p  data/setting1 \
#       -f  nucleosomes_vs_linkers_melanogaster.fas \
#       -o  data/setting1/pickle_H
#
# Output files (written to -o directory):
#   one_hot_nuc.pickle               — list of (147, 4)  arrays
#   one_hot_link.pickle              — list of (147, 4)  arrays
#   three_mer_one_hot_nuc.pickle     — list of (145, 12) arrays  ← model input
#   three_mer_one_hot_link.pickle    — list of (145, 12) arrays  ← model input
#
# Encoding details:
#   one_hot:             each base → 4-dim vector  (A,C,G,T)
#   three_mer_one_hot:   each overlapping 3-mer → 12-dim vector (3 × 4-dim)
#                        a 147 bp sequence yields 147-3+1 = 145 position rows

import os
import argparse
import pickle
import numpy as np
from Bio import SeqIO

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Encode FASTA sequences to pickle")
parser.add_argument("-p",  "--path",    dest="path",    type=str,
                    default=r"D:\data\setting1",      help="Directory containing FASTA file")
parser.add_argument("-f",  "--fas",     dest="fasName", type=str,
                    default="nucleosomes_vs_linkers_elegans.fas")
parser.add_argument("-o",  "--out",     dest="outDir",  type=str,
                    default=r"D:\data\setting1\pickle_H", help="Output directory for pickles")
parser.add_argument("-n",  "--nuc",     dest="nucPickle",  type=str, default="nuc.pickle")
parser.add_argument("-l",  "--lin",     dest="linkPickle", type=str, default="link.pickle")
args = parser.parse_args()

inPath     = args.path
outPath    = args.outDir
fasName    = args.fasName
nucPickle  = args.nucPickle
linkPickle = args.linkPickle

os.makedirs(outPath, exist_ok=True)

# ── Encoding functions ────────────────────────────────────────────────────────

BASE_MAP = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}

def one_hot(sequence: str) -> np.ndarray:
    """
    Encode a DNA sequence into a (len, 4) one-hot matrix.
    Unknown bases map to [0, 0, 0, 0].
    """
    encoded = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        encoded[i] = BASE_MAP.get(base, [0, 0, 0, 0])
    return encoded


def three_mer_one_hot(sequence: str) -> np.ndarray:
    """
    Encode overlapping 3-mers of a DNA sequence into a (len-2, 12) matrix.

    Each 3-mer is represented by concatenating three base one-hot vectors:
        A -> [1,0,0,0]
        C -> [0,1,0,0]
        G -> [0,0,1,0]
        T -> [0,0,0,1]
        N or unknown -> [0,0,0,0]

    Example:
        sequence length = 147
        output shape     = (145, 12)

    Returns:
        np.ndarray of shape (num_3mers, 12), dtype=float32
    """

    base_map = {
        'A': np.array([1, 0, 0, 0], dtype=np.float32),
        'C': np.array([0, 1, 0, 0], dtype=np.float32),
        'G': np.array([0, 0, 1, 0], dtype=np.float32),
        'T': np.array([0, 0, 0, 1], dtype=np.float32),
        'N': np.array([0, 0, 0, 0], dtype=np.float32)
    }

    seq = sequence.upper()
    seq_len = len(seq)
    num_3mers = seq_len - 2  # equivalent to len - 3 + 1

    if num_3mers <= 0:
        return np.empty((0, 12), dtype=np.float32)

    encoded = np.empty((num_3mers, 12), dtype=np.float32)

    for i in range(num_3mers):
        three_mer = seq[i:i+3]

        encoded[i] = np.concatenate([
            base_map.get(base, base_map['N'])
            for base in three_mer
        ])

    return encoded


# ── Parse FASTA and encode ────────────────────────────────────────────────────
nuc_oh_list   = []
link_oh_list  = []
nuc_tmoh_list = []
link_tmoh_list= []

fasta_path = os.path.join(inPath, fasName)
print(f"[INFO] Parsing {fasta_path} …")

n_nuc, n_link = 0, 0
for record in SeqIO.parse(fasta_path, "fasta"):
    seq  = str(record.seq)
    name = record.id.lower()

    if "nucleosomal" in name:
        nuc_oh_list.append(one_hot(seq))
        nuc_tmoh_list.append(three_mer_one_hot(seq))
        n_nuc += 1
    else:
        link_oh_list.append(one_hot(seq))
        link_tmoh_list.append(three_mer_one_hot(seq))
        n_link += 1

print(f"[INFO] Parsed {n_nuc} nucleosomal and {n_link} linker sequences.")

# Sanity checks
assert n_nuc  > 0, "No nucleosomal sequences found — check FASTA header naming."
assert n_link > 0, "No linker sequences found — check FASTA header naming."

sample_nuc  = nuc_tmoh_list[0]
sample_link = link_tmoh_list[0]
print(f"[INFO] Sample nucleosomal encoding shape : {sample_nuc.shape}  (expect 145,12 for 147 bp)")
print(f"[INFO] Sample linker encoding shape      : {sample_link.shape}")

# ── Save pickles ──────────────────────────────────────────────────────────────
def save_pickle(obj, filepath):
    with open(filepath, "wb") as fp:
        pickle.dump(obj, fp)
    print(f"  Saved → {filepath}  ({len(obj)} samples)")

save_pickle(nuc_oh_list,    os.path.join(outPath, f"one_hot_{nucPickle}"))
save_pickle(link_oh_list,   os.path.join(outPath, f"one_hot_{linkPickle}"))
save_pickle(nuc_tmoh_list,  os.path.join(outPath, f"three_mer_one_hot_{nucPickle}"))
save_pickle(link_tmoh_list, os.path.join(outPath, f"three_mer_one_hot_{linkPickle}"))

print("[INFO] Encoding complete.")