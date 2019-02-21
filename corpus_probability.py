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

def generate_jdd(cc_pair: dict) -> list:
    jdd = []
    for key in cc_pair.keys():
        for item in cc_pair[key]:
            jdd.append([item, key])
    return jdd

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

    pprint(cc_pair)

    jdd = np.asarray(generate_jdd(cc_pair))
    jdd = pd.DataFrame({'context': jdd[:, 0], 'center': jdd[:, 1]})
    print("Joint Distribution Table")
    print(jdd)

if __name__ == "__main__":
    main()
