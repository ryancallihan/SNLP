"""
SNLP Assignment 2
UPDATED
Submission from:
Ryan Callihan | 4076629
Mai Mayeg | 4053079
"""
import sys
from keras.callbacks import CSVLogger
import conllu as l
import transition as t
from keras.layers import Dense, Embedding, SimpleRNN
from keras.models import Sequential
import numpy as np
from collections import namedtuple

Token = namedtuple(
    'Token', "tid, form lemma pos xpos feats head deprel deps misc children")


class Numberer:
    """
    Class to turn elements into indices and back again
    Class taken from Dr. Daniel De Kok.
    """

    def __init__(self):
        self.v2n = dict()
        self.n2v = list()
        self.start_idx = 1

    def number(self, value, add_if_absent=True):
        n = self.v2n.get(value)

        if n is None:
            if add_if_absent:
                n = len(self.n2v) + self.start_idx
                self.v2n[value] = n
                self.n2v.append(value)
            else:
                return 0

        return n

    def value(self, number):
        return self.n2v[number - 1]

    def max_number(self):
        return len(self.n2v) + 1


def extract_feat_helper(config, sent, i):
    """
    Exercise 1
    Helper class for extract_feat_label
    :param config:
    :param sent:
    :param i:
    :return:
    """
    forms = []
    pos = []
    if i < len(config.stack):
        forms.append(sent.form[config.stack_nth(i)])
        pos.append(sent.upostag[config.stack_nth(i)])
    else:
        forms.append('None')
        pos.append('None')
    if i < len(config.input):
        forms.append(sent.form[config.input_nth(i)])
        pos.append(sent.upostag[config.input_nth(i)])
    else:
        forms.append('None')
        pos.append('None')
    return forms, pos


def extract_feat_label(s, include_forms=False, include_pos=True, stack_size=0, proj=False, lazy=True):
    '''
    Exercise 1
    Extracts features and labels from conllu sentences

    Can choose to include forms and/or POS

    :param s: Conllu stence
    :param include_forms: Include forms as feature
    :param include_pos: Include POS as feature
    :param stack_size: Size of stack
    :param proj:
    :param lazy:
    :return:
    '''
    o = t.Oracle(s, proj, lazy)
    c = t.Config(s)
    labels = []
    features = []
    while not c.is_terminal():
        act, arg = o.predict(c)
        labels.append((str(act) + "_" + str(arg)))
        feats = []
        for i in range(stack_size, 3):
            form, pos = extract_feat_helper(c, s, i)
            if include_pos:
                feats.extend(pos)
            if include_forms:
                feats.extend(form)
        features.append(feats)
        assert c.doable(act)
        getattr(c, act)(arg)
    return labels, features


def recode(elements, indices, is_feature=False, train=False):
    """
    Exercise 2
    Uses Numberer class to turn input into indices
    :param elements: Input
    :param indices: Numberer class indices
    :param is_feature: Checks for features of labels
    :param train:
    :return:
    """
    int_lex = list()

    for step in elements:
        if is_feature:
            int_idx = []
            for item in step:
                int_idx.append(indices.number(item, add_if_absent=train))
            int_lex.append(int_idx)
        else:
            int_lex.append(indices.number(step, add_if_absent=train))
    return int_lex


def model(max_x, output, input_dim):
    """
    Exercise 2
    :param output:
    :param max_x: Length of unique features
    :param input_dim: Starting dimension of input
    :return:
    """
    m = Sequential()
    m.add(Embedding(max_x, 128, input_length=input_dim))
    m.add(SimpleRNN(128, dropout=0.5, recurrent_dropout=0.2, activation='tanh'))
    m.add(Dense(output, activation='softmax'))
    m.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m


def get_predicted_labels(model, tst_x, n_y):
    """
    Exercise 2
    Gets predicted labels as tuples
    :param n_y:
    :param model: Pre-trained model
    :param tst_x: Testing features
    :return: Returns list actions as tuple
    """
    tst_pred = model.predict(tst_x)
    pred_conv = []
    for i in range(len(tst_pred)):
        sorted_actions = [n_y.value(act).split("_") for act in np.argsort(tst_pred[i])[::-1]]
        pred_conv.append(sorted_actions)
    return pred_conv


def parse(sentence, predicted_labels):
    """
    Exercise 3
    Produces a one best dependency tree using the predicted labels.
    :param sentence:
    :param predicted_labels:
    :return:
    """
    c = t.Config(sentence)

    for label in predicted_labels:
        for action, argument in label:
            if not c.is_terminal():
                if not c.doable(action):
                    continue
                else:
                    getattr(c, action)(argument)
                    break
    return c.finish()


def dep_scores(out, gs):
    """
    Produces dependency attachment scores.
    :param out:
    :param gs:
    :return:
    """

    if len(out) != len(gs):
        print("The number of sentences differ!")
        sys.exit(-1)

    arcs_lmatch_w = 0
    arcs_umatch_w = 0
    arcs_total = 0
    for i in range(len(out)):
        sent_out = out[i]
        sent_gs = gs[i]

        arcs_lmatch_sent = 0
        arcs_umatch_sent = 0
        ntokens = len(sent_out) - 1
        for j in range(1, len(sent_out)):
            if sent_out[j].head == sent_gs[j].head:
                arcs_umatch_sent += 1
                if sent_out[j].deprel == sent_gs[j].deprel:
                    arcs_lmatch_sent += 1
        arcs_total += ntokens
        arcs_lmatch_w += arcs_lmatch_sent
        arcs_umatch_w += arcs_umatch_sent

    print("UAS: {:.2f}\tLAS: {:.2f}".format(
        100 * arcs_umatch_w / arcs_total,
        100 * arcs_lmatch_w / arcs_total))


def read_conllu(fname=None, fp=sys.stdin, mark_children=False):
    """
    Reads conllu treebank
    :param fname:
    :param fp:
    :param mark_children:
    :return:
    """
    if fname is not None:
        fp = open(fname, 'r', encoding="utf-8")

    treebank = []
    sent_start = True
    for line in fp:
        if line.startswith('#'):
            continue
        line = line.strip()

        if len(line) == 0 and not sent_start:
            if mark_children:
                for tok in sent:
                    if tok.head is not None:
                        hd = sent[tok.head]
                        hd.children.append(tok.tid)
            treebank.append(sent)
            sent_start = True
            continue

        if mark_children:
            chi = []
        else:
            chi = None

        if sent_start:
            sent = [Token(
                0, "_", "root", "_", "_", "_", None, "_", "_", "_", chi)]
            sent_start = False

        (tid, form, lemma, pos, xpos, feats, head, deprel, deps, misc) = \
            line.strip().split('\t')
        if head == "":
            head = 0
        if "-" in tid:
            continue

        sent.append(Token(int(tid),
                          form,
                          lemma,
                          pos,
                          xpos,
                          feats,
                          int(head),
                          deprel.split(":")[0],
                          deps,
                          misc,
                          chi))
    return treebank


if __name__ == '__main__':

    """
    Exercise 4
    """

    """
    Loads conllu sentences
    """

    print("LOADING SENTENCES>>>>>")
    sents_trn = list(l.load('UD_English/en-ud-train.conllu'))  # Loads training sentences
    sents_tst = list(l.load('UD_English/en-ud-dev.conllu'))  # Loads all sentences

    features_trn = list()
    labels_trn = list()
    for sent in sents_trn:  # extracts features (forms and pos)
        label, feats = extract_feat_label(sent, include_forms=True)
        features_trn.extend(feats)
        labels_trn.extend(label)

    features_tst = list()
    labels_tst = list()
    for sent in sents_tst:  # extracts features (forms and pos)
        label, feats = extract_feat_label(sent, include_forms=True)
        features_tst.extend(feats)
        labels_tst.extend(label)

    print("ENCODING SENTENCES>>>>")
    n_x = Numberer()  # Encodes features
    trn_x = recode(features_trn, n_x, is_feature=True, train=True)
    tst_x = recode(features_tst, n_x, is_feature=True, train=True)

    n_y = Numberer()  # Encodes labels
    trn_y = recode(labels_trn, n_y, train=True)
    tst_y = recode(labels_tst, n_y, train=True)

    print("TRAINING SIZE:", np.array(trn_x).shape)
    print("TESTING SIZE:", np.array(tst_x).shape)

    """
    RNN Model
    """

    tsv_logger = CSVLogger('snlp2_training-data.tsv', append=True, separator='\t')
    m = model(n_x.max_number(), 1, np.array(trn_x).shape[1])
    m.fit(trn_x, trn_y, epochs=2, batch_size=128, verbose=2, shuffle=True, callbacks=[tsv_logger])
    score, acc = m.evaluate(tst_x, tst_y)  # Lets you know the true acc.

    print("\nSCORE: ", score)
    print("ACC: ", acc)

    print("BUILDING ONE BEST DEP TREEBANK>>>>")
    predicted_sents = []
    for i, sent in enumerate(sents_tst):
        labels, feats = extract_feat_label(sent, include_forms=True)
        x = recode(feats, n_x, is_feature=True)
        pred = get_predicted_labels(m, x, n_y)
        predicted_sents.append(parse(sent, predicted_labels=pred))

    l.save(predicted_sents, "UD_English/pred-dep-trees.conllu")

    """
    Get LAS|UAS
    """

    print("GETTING LAS | UAS SCORES>>>>")
    out_sents = read_conllu("UD_English/pred-dep-trees.conllu")
    gs_sents = read_conllu("UD_English/en-ud-dev.conllu")

    dep_scores(out_sents, gs_sents)

    """
    TRAINING SIZE: (411316, 12)
    TESTING SIZE: (50476, 12)
    Epoch 1/20
    267s - loss: 0.9071 - acc: 0.7488
    Epoch 2/20
    256s - loss: 0.5904 - acc: 0.8185
    Epoch 3/20
    256s - loss: 0.5309 - acc: 0.8365
    Epoch 4/20
    254s - loss: 0.5001 - acc: 0.8462
    Epoch 5/20
    257s - loss: 0.4798 - acc: 0.8521
    Epoch 6/20
    256s - loss: 0.4627 - acc: 0.8581
    Epoch 7/20
    256s - loss: 0.4497 - acc: 0.8616
    Epoch 8/20
    255s - loss: 0.4400 - acc: 0.8646
    Epoch 9/20
    256s - loss: 0.4317 - acc: 0.8671
    Epoch 10/20
    255s - loss: 0.4256 - acc: 0.8687
    Epoch 11/20
    256s - loss: 0.4179 - acc: 0.8712
    Epoch 12/20
    256s - loss: 0.4133 - acc: 0.8727
    Epoch 13/20
    258s - loss: 0.4086 - acc: 0.8741
    Epoch 14/20
    260s - loss: 0.4053 - acc: 0.8745
    Epoch 15/20
    256s - loss: 0.3997 - acc: 0.8762
    Epoch 16/20
    256s - loss: 0.3977 - acc: 0.8770
    Epoch 17/20
    256s - loss: 0.3941 - acc: 0.8786
    Epoch 18/20
    255s - loss: 0.3921 - acc: 0.8784
    Epoch 19/20
    255s - loss: 0.3895 - acc: 0.8795
    Epoch 20/20
    256s - loss: 0.3883 - acc: 0.8798
    SCORE:  0.539435010124
    ACC:  0.841548458678
    GETTING LAS | UAS SCORES>>>>
    UAS: 44.92	LAS: 37.72

    """
