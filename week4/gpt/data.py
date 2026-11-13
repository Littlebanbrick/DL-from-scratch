import torch


def load_text(path="data/input.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_tokenizer(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[int(i)] for i in ids)

    return stoi, itos, encode, decode


def load_split(text, ratio=0.9):
    stoi, itos, encode, decode = make_tokenizer(text)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(ratio * len(data))
    return data[:n], data[n:], stoi, itos, encode, decode


def get_batch(data, block_size, batch_size, device="cpu"):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
