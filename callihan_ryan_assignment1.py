#!/usr/bin/env python

import os
import pickle
import time
import numpy as np
import nltk
import callihan_ryan_assignment1 as snlp
from collections import Counter


def main():
    process_start = time.time()
    training_start = time.time()
    files_c = list()
    filenames_c = list()
    files_d = list()
    filenames_d = list()
    files_t = list()
    filenames_t = list()

    path = "/home/ryancallihan/Dropbox/school/snlp_s17/assignment1-data/"
    print("Loading training files")
    for file in sorted(os.listdir(path)):
        if file[0] == "d":
            files_d.append(snlp.tokenize(path.strip() + file.strip()))
            filenames_d.append(file[0:3])
        elif file[0] == "c":
            files_c.append(snlp.tokenize(path.strip() + file.strip()))
            filenames_c.append(file[0:3])
        else:
            files_t.append(snlp.tokenize(path.strip() + file.strip()))
            filenames_t.append(file[0:3])
    print("Training files loaded")

    """
    Train ngram models
    """
    print("Working on d ngrams")
    d = dict()
    for n in range(1, 4):
        print(n, "grams")
        d[n] = snlp.Ngram.train_model(n, files_d[1:], files_c[0])
        print("# grams: {} | optimal alpha: {}".format(len(d[n].ngram), d[n].alpha))
    snlp.save_object(d, "d.pkl")
    d = None

    print("Working on c ngrams")
    c = dict()
    for n in range(1, 4):
        print(n, "grams")
        c[n] = snlp.Ngram.train_model(n, files_c[2:], files_c[1])
        print("# grams: {} | optimal alpha: {}".format(len(c[n].ngram), c[n].alpha))
    snlp.save_object(c, "c.pkl")
    c = None

    """
    Train back-off models
    """
    print("Working on d backoff ngrams")
    d_backoff = snlp.BackoffNgram.train_model_backoff(files_d[2:])
    snlp.save_object(d_backoff, "d_backoff.pkl")
    print("# grams: {} | beta: {} | lambda: {} | token count: {}"
          .format(len(d_backoff.ngram), d_backoff.beta, d_backoff.lam, d_backoff.token_count))
    d_backoff = None

    print("Working on c backoff ngrams")
    c_backoff = snlp.BackoffNgram.train_model_backoff(files_c[2:])
    snlp.save_object(c_backoff, "c_backoff.pkl")
    print("# grams: {} | beta: {} | lambda: {} | token count: {}"
          .format(len(c_backoff.ngram), c_backoff.beta, c_backoff.lam, c_backoff.token_count))
    c_backoff = None
    print("Training time:", ((time.time() - training_start) / 60.0))

    """
    Get Perplexities
    """
    files = list()
    files.append(files_c[0])
    files.append(files_d[0])
    files.extend(files_t)
    filenames = list()
    filenames.append(filenames_c[0])
    filenames.append(filenames_d[0])
    filenames.extend(filenames_t)
    files_c = filenames_c = files_d = filenames_d = files_t = filenames_t = None

    pp_matrix = snlp.prob_test(files, filenames)
    snlp.save_object(pp_matrix, "pp_matrix.pkl")

    """
    Display data
    """
    snlp.print_prob_matrix(pp_matrix)
    model = pickle.load(open("c.pkl", "rb"))
    c1_alpha = model[1].alpha
    c2_alpha = model[2].alpha
    c3_alpha = model[3].alpha
    model = pickle.load(open("d.pkl", "rb"))
    d1_alpha = model[1].alpha
    d2_alpha = model[2].alpha
    d3_alpha = model[3].alpha
    model = pickle.load(open("c_backoff.pkl", "rb"))
    cb_lambda = model.lam
    cb_beta = model.beta
    model = pickle.load(open("d_backoff.pkl", "rb"))
    db_lambda = model.lam
    db_beta = model.beta
    print("ALPHA HYPERPARAM-\n C1:", c1_alpha, "C2:", c2_alpha, "C3:", c3_alpha, "\nD1:", d1_alpha, "D2:", d2_alpha,
          "D3:",
          d3_alpha, "\n\nGT DISCOUNT-\n C Tri:", cb_lambda, "C Bi:", cb_beta, "\nD Tri:", db_lambda, "D Bi:", db_beta)

    print("Entire process takes:", ((time.time() - process_start) / 60.0))


if __name__ == '__main__': main()

"""
Methods used by both ngram and backoff models.
"""


def tokenize(filename):
    tokenized = list()
    text = open(filename, "r").read()
    for sentence in nltk.sent_tokenize(text):
        tokenized.append(nltk.word_tokenize(sentence))
    return tokenized


def seq_to_ngrams(sentence, n):
    bos_str = ("<s>",) * (n - 1)
    eos_str = "<\s>"
    padded_seq = list(bos_str)
    padded_seq.extend(sentence)
    padded_seq.append(eos_str)
    return zip(*[padded_seq[i:] for i in range(1 + len(bos_str))])


def estimate_alpha(model, text):
    alpha = 0
    score = float("+inf")
    for i in np.arange(0.0, 1.1, 0.1):
        current = model.perplexity(text, alpha=i)
        if current > score:
            break
        else:
            alpha = i
            score = current
    return alpha


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    print("Saved: ", type(obj), " - as: ", filename)


def prob_test(files, filenames):
    pp_matrix = list()
    pp_matrix_files = list()
    pp_matrix_files.append("file")
    for filename in filenames:
        pp_matrix_files.append(filename)
    pp_matrix.append(pp_matrix_files)
    pp_matrix_col = list()

    for n in range(1, 4):
        for prob in ["l", "a"]:
            for author in ["c", "d"]:
                pp_matrix_col.append("{}-{}g-{}".format(author, n, prob))
                if author == "c":
                    model = pickle.load(open("c.pkl", "rb"))
                elif author == "d":
                    model = pickle.load(open("d.pkl", "rb"))
                print("Working on author:", author, "Prob:", prob, "Ngram:", n)

                for text in files:
                    pp = 0.0
                    if prob == "a":
                        pp = model[n].perplexity(text, alpha=model[n].alpha)
                    if prob == "l":
                        pp = model[n].perplexity(text)
                    pp_matrix_col.append(pp)
                pp_matrix.append(pp_matrix_col)
                pp_matrix_col = list()

    for author in ["c", "d"]:
        model = None
        pp_matrix_col = list()
        pp_matrix_col.append("{}-backoff".format(author))
        if author == "c":
            model = pickle.load(open("c_backoff.pkl", "rb"))
        elif author == "d":
            model = pickle.load(open("d_backoff.pkl", "rb"))
        print("Working on backoff author:", author, "lambda:", model.lam, "beta:", model.beta)
        for text in files:
            pp = model.perplexity(text)
            pp_matrix_col.append(pp)
        pp_matrix.append(pp_matrix_col)
    return pp_matrix


def print_prob_matrix(pp_matrix):
    for i in list(zip(*pp_matrix)):
        for j in i:
            if isinstance(j, np.float):
                print("%.1f\t" % j, end="")
            else:
                print(j, "\t", end="")
        print()


class Ngram():
    """
    Class for vanilla ngrams
    """

    def __init__(self, n):
        self.n = n
        self.ngram = Counter()
        self.prefix = Counter()
        self.types = set()
        self.alpha = float()

    def update(self, sequence):
        for sentence in sequence:
            self.types.update(sentence)
            for ngram in seq_to_ngrams(sentence, self.n):
                self.ngram[ngram] += 1
                self.prefix[ngram[:-1]] += 1

    def prob_mle(self, sentence):
        text_prob = 0.0
        for ng in seq_to_ngrams(sentence, self.n):
            ng_count = self.ngram[ng]
            if not ng_count:
                return float("-inf")
            else:
                text_prob += np.log2(ng_count / (self.prefix[ng[:-1]]))
        return text_prob

    def prob_add(self, sentence, alpha=1):
        text_prob = 0.0
        for ng in seq_to_ngrams(sentence, self.n):
            ng_count = self.ngram.get(ng, 0)
            text_prob += np.log2((ng_count + alpha) / (self.prefix[ng[:-1]] + (alpha * len(self.types))))
        return text_prob

    def perplexity(self, text, alpha=1):
        prob = 0.0
        token_count = 0
        for sentence in text:
            token_count += len(sentence) + 1
        for sent in text:
            prob += self.prob_add(sent, alpha=alpha)
        return 2 ** (-(1.0 / token_count) * prob)

    @staticmethod
    def train_model(n, training_files, alpha_training_file):
        model = Ngram(n)
        for file in training_files:
            model.update(file)
        model.alpha = snlp.estimate_alpha(model, alpha_training_file)
        return model


class BackoffNgram:
    """
    Back-off ngram models
    """

    def __init__(self):
        self.ngram = Counter()
        self.prefix = Counter()
        self.types = set()
        self.lam = float()
        self.beta = float()
        self.token_count = int()

    def update(self, sequence, n):
        if not isinstance(sequence, list):
            sequence = list(sequence)
        for sentence in sequence:
            self.types.update(sentence)
            for ng in seq_to_ngrams(sentence, n):
                self.ngram[ng] += 1
                self.prefix[ng[:-1]] += 1

    def backoff_model(self, sentence, token_count, alpha=1):
        text_prob = 0.0
        for ng in seq_to_ngrams(sentence, 3):
            if self.ngram[ng] > 0:
                text_prob += np.log2((1.0 - self.lam) * (self.ngram[ng] / self.prefix[ng[:-1]]))
            elif self.ngram[ng[1:3]] > 0:
                text_prob += np.log2(self.lam * (1.0 - self.beta) * (self.ngram[ng[1:3]] / self.prefix[ng[1:2]]))
            else:
                text_prob += np.log2(
                    self.lam * self.beta * ((self.ngram[ng[2:3]] + alpha) / (self.token_count + (alpha * len(self.types)))))
        return text_prob

    def perplexity(self, text, alpha=1):
        prob = 0.0
        token_count = 0
        for sentence in text:
            token_count += len(sentence) + 1
        for sent in text:
            prob += self.backoff_model(sent, token_count, alpha=alpha)
        return 2 ** (-(1.0 / token_count) * prob)

    @staticmethod
    def train_model_backoff(training_files):
        model = BackoffNgram()
        for n in reversed(range(1, 4)):
            for file in training_files:
                model.update(file, n)
        print("Finding lambda")
        model.lam = sum([len(k) == 3 and v <= 1 for k, v in model.ngram.items()]) / sum(
            [v for k, v in model.ngram.items() if len(k) == 3])
        print("Finding beta")
        model.beta = sum([len(k) == 2 and v <= 1 for k, v in model.ngram.items()]) / sum(
            [v for k, v in model.ngram.items() if len(k) == 2])
        model.token_count = sum([v for k, v in model.ngram.items() if len(k) == 1])
        return model


"""
-The bigram model with additive smoothing worked well for discriminating both authors

-According to the model, the test files T2, T3, T5 belong to author c

-According to the model, the test files T1, T4 belong to author d

Author "c" - Wilkie Collins
Author "d" - Charles Dickens

file 	c-1g-l 	d-1g-l 	c-1g-a 	d-1g-a 	c-2g-l 	d-2g-l 	c-2g-a 	d-2g-a 	c-3g-l 	d-3g-l 	c-3g-a 	d-3g-a 	c-backoff 	d-backoff 	
c00 	763.6	807.7	763.6	807.7	1053.9	1428.0	465.2	579.5	8071.1	12923.3	3208.6	4998.6	345.6	387.6	
d00 	929.5	743.6	929.5	743.6	1462.1	1217.0	682.3	486.1	10885.5	10644.9	4885.6	3869.0	594.2	280.0	
t01 	876.4	699.9	876.4	699.9	1320.3	1098.2	616.4	435.6	9985.5	9842.1	4433.2	3502.7	545.9	254.9	
t02 	691.2	736.5	691.2	736.5	875.1	1179.0	407.2	480.0	6518.5	10512.9	2676.9	3989.9	333.7	325.0	
t03 	674.6	749.8	674.6	749.8	875.3	1284.4	376.0	511.6	6911.4	11908.0	2615.1	4502.0	259.8	325.8	
t04 	690.9	675.1	690.9	675.1	990.0	1130.6	474.6	457.4	7669.1	10461.7	3363.0	3876.2	425.0	299.2	
t05 	567.2	661.9	567.2	661.9	601.2	979.6	249.1	379.5	4651.4	9230.3	1534.6	3179.2	155.2	240.0	
t06 	857.8	807.4	857.8	807.4	1274.3	1441.4	645.8	612.7	9007.0	12368.2	4405.4	5193.3	645.9	450.7	
t07 	1972.8	1870.2	1972.8	1870.2	2625.1	2993.4	1529.1	1522.6	15242.7	20890.1	8327.4	10179.4	2660.6	1814.3	
t08 	1029.1	1055.6	1029.1	1055.6	1439.7	1756.2	666.0	730.8	11082.4	15693.4	4986.2	6484.4	710.2	608.1	
ALPHA HYPERPARAM-
 C1: 1.0 C2: 0.1 C3: 0.1 
D1: 1.0 D2: 0.1 D3: 0.1 

GT DISCOUNT-
 C Tri: 0.34961001764416205 C Bi: 0.09533809880289108 
D Tri: 0.39634027967019725 D Bi: 0.1142963314963414
Entire process takes: 5.8020714044570925

file | author
T01 - Dickens
T02 - Dickens
T03 - Collins
T04 - Dickens
T05 - Collins
T06 - Mark Twain
T07 - Lewis Carroll
T08 - Jane Austen

"""