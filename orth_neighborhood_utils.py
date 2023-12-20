import string
import numpy as np
import itertools
import matplotlib.pyplot as plt
import random

def wordlist_to_orth(wordlist):
    # takes in list of K words, outputs a 104 x K "spelling" array
    alphabet = string.ascii_lowercase
    wordids = np.array([np.array([alphabet.index(L) for L in word]) for word in wordlist])
    onehots = np.zeros((len(wordlist), len(wordlist[0])*len(alphabet)))
    for i in range(len(wordlist)):
        indices = np.add(wordids[i],np.array([0,1,2,3])*26)
        onehots[i,:][indices] = 1
    return onehots.T

def wordlist_to_ctx(word_stimlist, lexicon, cloze = 0.99, preact_resource = 2.0):
    # takes in a list of K words, assigns a high probability (p = cloze) to each of them in separate trials
    # preact_resource is the value that the pseudoprobability distribution sums to
    # seems like I don't need a word_stimlist.
    lexicon_indices = np.array([lexicon.index(w) for w in word_stimlist])
    low_prob = np.multiply(np.ones((len(lexicon),len(word_stimlist))), preact_resource / (len(lexicon) - 1))* (1 - cloze)
    low_prob[lexicon_indices, np.arange(len(lexicon_indices))] = preact_resource*cloze
    return low_prob

def orth_to_wordlist(onehots):
    # change an N x 104 matrix of one-hot encodings to a list of words
    alphabet = string.ascii_lowercase
    NumWords = onehots.shape[0]
    LettersPerWord = int(onehots.shape[1]/len(alphabet))
    matrix_perword = onehots.reshape((NumWords,LettersPerWord,26))
    wordlist = []
    for w in range(NumWords):
        letterlist = [alphabet[i] for i in np.where(matrix_perword[w][:])[-1]]
        wordlist.append(''.join(letterlist))
    return wordlist