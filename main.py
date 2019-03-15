import sys
import torch
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F
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

    # Return center,context pairs
    return pairs

def get_idxpairs(cc_pair: dict, w2idx: list) -> list:
    """
    The generate_center_context_pair gives a dictionary like:
    {'center word 1': ['contextword1', 'contextword2', '...']
     'centerword2': ['contextword1', 'contextword2', '...']}
    But the code from the blog needs cc_pair like:
    [['centerword1', 'contextword1'],
     ['centerword1', 'contextword2'], ...]
    So this part changes from the former format to the latter
    """
    idx_pairs = []
    for center in cc_pair.keys():
        for context in cc_pair[center]:
            idx_pairs.append([w2idx[center], w2idx[context]])
    return idx_pairs

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

# Neural Net functions
def get_input_layer(word_idx, vocab_size):
    x = torch.zeros(vocab_size).float()
    x[word_idx] = 1.0
    return x

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

def experiments():
    """This function has all the codes that were used to experiment and understand the project.

    All of this code used to be in the main function. All of it is self
    written."""
    tokens = tokenize(corpus)
    cc_pair = generate_center_context_pair(tokens, 2)
    pprint(corpus)

    global jdt
    jdt = np.asarray(generate_jdt(cc_pair))
    jdt = pd.DataFrame({'center': jdt[:, 0], 'context': jdt[:, 1]})
    print("Joint Distribution Table")
    print(jdt)

    cc_pair_counts = all_p_of_context_given_center(jdt)
    pprint(cc_pair_counts)

def main():
    tokens = tokenize(corpus)
    vocabulary = set(sum(tokens, [])) # sum() flattens the 2d list
    vocab_size = len(vocabulary)
    cc_pair = generate_center_context_pair(tokens, 2)
    # pprint(cc_pair)

    word2idx = word2index(tokens)
    idx2word = {key: val for (val, key) in word2idx.items()}
    print(word2idx)
    print(idx2word)

    idx_pairs = get_idxpairs(cc_pair, word2idx)
    idx_pairs = np.array(idx_pairs)

    embedding_dims = 5
    W1 = Variable(torch.randn(embedding_dims, vocab_size).float(),
            requires_grad=True)
    W2 = Variable(torch.randn(vocab_size, embedding_dims).float(),
            requires_grad=True)
    max_iter = int(sys.argv[1])
    learning_rate = 0.001

    for i in range(max_iter):
        loss_val = 0
        for data, target in idx_pairs:
            x = Variable(get_input_layer(data, vocab_size)).float()
            y_true = Variable(torch.from_numpy(np.array([target])).long())

            z1 = torch.matmul(W1, x)
            z2 = torch.matmul(W2, z1)

            log_softmax = F.log_softmax(z2, dim=0)

            loss = F.nll_loss(log_softmax.view(1, -1), y_true)
            loss_val += loss.item()
            loss.backward()
            W1.data -= learning_rate * W1.grad.data
            W2.data -= learning_rate * W2.grad.data

            W1.grad.data.zero_()
            W2.grad.data.zero_()
        if i % 10 == 0:
            print(f"Loss at iter {i}: {loss_val/len(idx_pairs)}")

if __name__ == "__main__":
    main()
