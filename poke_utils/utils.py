from os import listdir
from os.path import isfile, join
from collections import Counter
from math import sqrt
import numpy as np
from difflib import SequenceMatcher
import textdistance

def list_poke_names(path = 'Pokemon Figures'):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    pokes = [file.split('.')[0] for file in files]
    return pokes

def list_poke_names(path = 'Pokemon Figures'):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    pokes = [file.split('.')[0] for file in files]
    return pokes

def word2vec(word):
    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c*c for c in cw.values()))

    # return a tuple
    return cw, sw, lw

def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]

def similarity_word(w1, w2):
    return SequenceMatcher(None, w1, w2).ratio()

def most_similar(name, name_list, metric = 'cosin'):
    if metric == 'cosin':
        distances = [cosdis(word2vec(name),word2vec(other_name)) for other_name in name_list]
    elif metric == 'sequence':
        distances = [similarity_word(name,other_name) for other_name in name_list]
    elif metric == 'levenshtein':
        distances = [textdistance.levenshtein(name,other_name) for other_name in name_list]
    elif metric == 'jaro_winkler':
        distances = [textdistance.jaro_winkler(name,other_name) for other_name in name_list]
    elif metric == 'damerau_levenshtein':
        distances = [textdistance.damerau_levenshtein(name,other_name) for other_name in name_list]
        
    return name_list[np.argmax(distances)]