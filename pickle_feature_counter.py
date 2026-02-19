import pickle

with open('data/DatasetNup_1/encoded_melanogaster/three_mer_one_hot_nuc.pickle', 'rb') as f:
  data = pickle.load(f)

for idx, seq in enumerate(data):
  print(f"Sequence {idx+1}: {len(seq)} base pairs")