import numpy as np
import pandas as pd
from pprint import pprint

# Function defs

def tokenize(corpus : str) -> list:
    tokens = []
    for sentence in corpus:
        tokens.append(sentence.split())
    return tokens

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

    global jdt
    jdt = np.asarray(generate_jdt(cc_pair))
    jdt = pd.DataFrame({'center': jdt[:, 0], 'context': jdt[:, 1]})
    print("Joint Distribution Table")
    print(jdt)

    cc_pair_counts = all_p_of_context_given_center(jdt)
    pprint(cc_pair_counts)

if __name__ == "__main__":
    main()
