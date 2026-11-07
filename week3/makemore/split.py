import random

SEED = 0
RATIO = (0.7, 0.2, 0.1)


def load_names(path="data/names.txt"):
    with open(path, "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]
    return sorted(set(rows))


def get_splits(path="data/names.txt", seed=SEED, ratio=RATIO):
    names = load_names(path)
    random.Random(seed).shuffle(names)

    n = len(names)
    n_train = int(ratio[0] * n)
    n_dev = int(ratio[1] * n)

    train = names[:n_train]
    dev = names[n_train:n_train + n_dev]
    test = names[n_train + n_dev:]
    return train, dev, test


if __name__ == "__main__":
    train, dev, test = get_splits()
    all_names = train + dev + test

    for tag, part in (("train", train), ("dev", dev), ("test", test)):
        pairs = sum(len(n) + 1 for n in part)
        print(f"{tag:5s}  names {len(part):6d}  pairs {pairs:7d}  ({len(part) / len(all_names):.1%})")

    # 确认没有重叠
    assert not (set(train) & set(dev)), "train/dev overlap"
    assert not (set(train) & set(test)), "train/test overlap"
    assert not (set(dev) & set(test)), "dev/test overlap"
    assert len(all_names) == len(set(all_names)), "duplicates in source"
    print(f"\ntotal names {len(all_names)}   no overlap, seed={SEED}")
