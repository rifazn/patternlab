import numpy as np
import pandas as pd
from pprint import pprint

# Function defs

def tokenize(corpus : str) -> list:
    tokens = []
    for sentence in corpus:
        tokens.append(sentence.split())
    return tokens

def word2index(tokens):
    vocabulary = []
    for sentence in tokens:
        for token in sentence:
            if token not in vocabulary:
                vocabulary.append(token)
    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}    
    return word2idx

def generate_center_context_pair(tokens, window: int) -> dict:
    pairs = dict()
    for row in tokens:
        for idx, center_word in enumerate(row):
            pairs.setdefault(center_word, [])
            for i in range(idx - window, idx + window + 1):
                if (i >= 0 and i != idx and i < len(row)):
                    pairs[center_word].append(row[i])
    return pairs

def generate_jdt(cc_pair: dict) -> list:
    jdt = []
    for center in cc_pair.keys():
        for context in cc_pair[center]:
            jdt.append([center, context])
    return jdt

def all_p_of_context_given_center(joint_distrib_table: pd.DataFrame):
    counts = joint_distrib_table.groupby(['center', 'context']).size()
    counts = counts.to_dict()

    # Denominator for the probability
    total = joint_distrib_table.groupby('center').size()
    total = total.to_dict()

    for center in total.keys():
        for k in list(counts.keys()):
            if k[0] is center:
                counts[k] = [counts[k]]
                counts[k].append(total[center])

    return counts

corpus = [
        "he is a king",
        "she is a queen",
        "he is a man",
        "she is a woman",
        "warsaw is poland capital",
        "berlin is germany capital",
        "paris is france capital",
        # "Sxi este juna kaj bela",
]

def main():
    pprint(corpus)

    tokens = tokenize(corpus)
    cc_pair = generate_center_context_pair(tokens, 2)

    # pprint(cc_pair)

    word2idx = word2index(tokens)
    idx2word = {key: val for (val, key) in word2idx.items()}
    print(word2idx)
    print(idx2word)

    global jdt
    jdt = np.asarray(generate_jdt(cc_pair))
    jdt = pd.DataFrame({'center': jdt[:, 0], 'context': jdt[:, 1]})
    print("Joint Distribution Table")
    print(jdt)

    cc_pair_counts = all_p_of_context_given_center(jdt)
    pprint(cc_pair_counts)

if __name__ == "__main__":
    main()
