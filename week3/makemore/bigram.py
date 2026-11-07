import numpy as np

# Encoding a-z and '.' into numbers
stoi = {
    '.': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,
    'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17,
    'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26
}
itos = {i: c for c, i in stoi.items()}

with open("data/names.txt", "r", encoding="utf-8") as f:
    names = [line.strip() for line in f]


counts = np.zeros((27, 27), dtype=int)

for name in names:
    first = name[0]
    last = name[-1]
    counts[stoi['.']][stoi[first]] += 1
    counts[stoi[last]][stoi['.']] += 1

    for i in range(len(name)-1):
        counts[stoi[name[i]]][stoi[name[i+1]]] += 1

print(f"names          {len(names)}")
print(f"counts shape   {counts.shape}")
print(f"counts sum     {counts.sum()}   (expected {sum(len(n)+1 for n in names)})")
print(f"row '.'        {counts[0].sum()}   col '.' {counts[:,0].sum()}")


P = counts / counts.sum(1, keepdims=True).clip(min=1)

nll_sum, n_pairs = 0.0, 0
for name in names:
    seq = ['.'] + list(name) + ['.']
    for a, b in zip(seq, seq[1:]):
        nll_sum += -np.log(P[stoi[a], stoi[b]])
        n_pairs += 1

print(f"pairs          {n_pairs}")
print(f"NLL            {nll_sum / n_pairs:.4f}")    # NLL: negative log likelihood
