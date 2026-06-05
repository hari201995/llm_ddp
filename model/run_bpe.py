import pickle
from cs336_basics.train_bpe import train_bpe_fn

vocab, merges = train_bpe_fn(
    "/Users/hari/Documents/backups/Datasets/owt_train.txt",
    32000,
    ["<|endoftext|>"],
)
pickle.dump(vocab, open("../tokenizer/owt_vocab.pkl", "wb"))
pickle.dump(merges, open("../tokenizer/owt_merges.pkl", "wb"))
print("Done")
