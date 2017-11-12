'''
Ryan Callihan
Mai Mayeg
SNLP Assignment 2
'''
import sys

import conllu as l
import transition as t
from keras.layers import Dense, Embedding, Bidirectional, LSTM, SimpleRNN
from keras.models import Sequential, model_from_json
import numpy as np
from collections import namedtuple
import sklearn.metrics as mt
from keras.preprocessing.text import Tokenizer
import pickle

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
        return self.n2v[number - 1]  # TODO check index

    def max_number(self):
        return len(self.n2v) + 1


def load_trees(file_name):
    sents = l.load(file_name)
    return sents


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
    '''
    Exercise 2
    Uses Numberer class to turn input into indices
    :param elements: Input
    :param indices: Numberer class indices
    :param is_feature: Checks for features of labels
    :param train:
    :return:
    '''
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


def model(trn_x, trn_y, max_x, max_y, input_dim):
    """
    Exercise 2
    Keras Bidirection LSTM classifier
    :param trn_x: Training features
    :param trn_y: Training labels
    :param max_x: Length of unique features
    :param max_y: Length of unique labels
    :param input_dim: Starting dimension of input
    :return:
    """
    m = Sequential()
    # m.add(Embedding(max_x, 128, input_length=input_dim))
    # m.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.05)))
    # m.add(Dense(max_y, activation='softmax'))
    # m.compile(optimizer='adam',
    #           loss='sparse_categorical_crossentropy',
    #           metrics=['accuracy'])

    embedding_size = 100
    m.add(Embedding(input_dim=max_x,
                    output_dim=embedding_size,
                    input_length=input_dim))
    m.add(SimpleRNN(embedding_size,
                    dropout=0.5,
                    recurrent_dropout=0.1,
                    activation="tanh"))

    m.add(Dense(max_y,
                activation="softmax"))

    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

    return m


def get_predicted_labels(model, tst_x, n_y):
    """
    Exercise 2
    Gets predicted labels as tuples
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
    print(len(out))
    print(len(gs))

    if len(out) != len(gs):
        print("The number of sentences differ!")
        sys.exit(-1)

    # arcs_lmatch_s = 0
    # arcs_umatch_s = 0

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


    # if __name__ == '__main__':


"""
Exercise 4
"""

sents_trn = list(l.load('UD_English/en-ud-train.conllu'))  # Loads training sentences
sents_tst = list(l.load('UD_English/en-ud-dev.conllu'))  # Loads all sentences
# sents_tst = read_conllu('UD_English/en-ud-dev.conllu')

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


n_x = Numberer()  # Encodes features
trn_x = recode(features_trn, n_x, is_feature=True, train=True)
tst_x = recode(features_tst, n_x, is_feature=True, train=True)

n_y = Numberer()  # Encodes labels
trn_y = recode(labels_trn, n_y, train=True)
tst_y = recode(labels_tst, n_y, train=True)

print("TRAINING SIZE:", np.array(trn_x).shape)
print("TESTING SIZE:", np.array(tst_x).shape)

input_dim = np.array(trn_x).shape[1]  # gets input dimentions, which is important for the embedding
m = model(trn_x, trn_y, n_x.max_number(), n_y.max_number(), input_dim)

# You can also select batch size here
m.fit(trn_x, trn_y, epochs=5, batch_size=128, verbose=2)

score, acc = m.evaluate(tst_x, tst_y)  # Lets you know the true acc.

print("\nSCORE: ", score)
print("ACC: ", acc)

"""
Saves model to disk.
"""
model_json = m.to_json()

with open("snlp-ud-lstm.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model to disk.")
m.save_weights("snlp-ud-lstm.h5")
print("Saved weights to disk.")
pickle.dump(n_x, open("feature_encoder.pkl", "wb"))
pickle.dump(n_y, open("label_encoder.pkl", "wb"))

print("Finished LSTM")


# """
# For loading pretrained models
# """
# # Opens pretrained model
# json_file = open("snlp-ud-lstm.json", 'r')
#
# # Reads pretrained model
# m_loaded = json_file.read()
# json_file.close()
#
# # Model is "m"
# m = model_from_json(m_loaded)
#
# # Loads pretrained weights
# m.load_weights("snlp-ud-lstm.h5")
# print("Loaded model from disk!")
#
# # Compiles model
# m.compile(optimizer='adam',
#           loss='categorical_crossentropy',
#           metrics=['accuracy'])

predicted_sents = []
for i, sent in enumerate(sents_tst):
    labels, feats = extract_feat_label(sent, include_forms=True)
    x = recode(feats, n_x, is_feature=True)
    pred = get_predicted_labels(m, x, n_y)
    predicted_sents.append(parse(sent, predicted_labels=pred))

l.save(predicted_sents, "UD_English/pred-dep-trees.conllu")

out_sents = read_conllu("UD_English/pred-dep-trees.conllu")
gs_sents = read_conllu("UD_English/en-ud-dev.conllu")

dep_scores(out_sents, gs_sents)

#
# """
# For loading pretrained models
# """
# # Opens pretrained model
# json_file = open("snlp-ud-lstm.json", 'r')
#
# # Reads pretrained model
# m_loaded = json_file.read()
# json_file.close()
#
# # Model is "m"
# m = model_from_json(m_loaded)
#
# # Loads pretrained weights
# m.load_weights("snlp-ud-lstm.h5")
# print("Loaded model from disk!")
#
# # Compiles model
# m.compile(optimizer='adam',
#           loss='categorical_crossentropy',
#           metrics=['accuracy'])
#
# # Loads the encoder for labels
# n_y = pickle.load(open("label_encoder.pkl", "rb"))
#
# # Loads encoder for features
# n_x = pickle.load(open("feature_encoder.pkl", "rb"))
#
# sents_tst = list(l.load('UD_English/en-ud-dev.conllu'))
#
# # Test sentence
# test_label, test_steps = extract_feat_label(sents_tst[2], include_forms=True)
#
# test_steps = recode(test_steps, n_x, is_feature=True)

# This gets the output from loaded sentences when passed through the model.
# pred = get_predicted_labels(m, test_steps, n_y)
#
# for i, p in enumerate(pred[:30]):
#     print(i, "PRED:", p, "| Actual:", test_label[i])

"""
888s - loss: 0.8376 - acc: 0.7658
Epoch 2/50
861s - loss: 0.4580 - acc: 0.8563
Epoch 3/50
^T883s - loss: 0.3667 - acc: 0.8861
Epoch 4/50
884s - loss: 0.3076 - acc: 0.9047
Epoch 5/50
946s - loss: 0.2654 - acc: 0.9177
Epoch 6/50
1072s - loss: 0.2325 - acc: 0.9276
Epoch 7/50
1010s - loss: 0.2066 - acc: 0.9351
Epoch 8/50
1005s - loss: 0.1850 - acc: 0.9415
Epoch 9/50
1089s - loss: 0.1671 - acc: 0.9468
Epoch 10/50
1052s - loss: 0.1521 - acc: 0.9515
Epoch 11/50
837s - loss: 0.1398 - acc: 0.9550
Epoch 12/50
835s - loss: 0.1287 - acc: 0.9583
Epoch 13/50
834s - loss: 0.1189 - acc: 0.9614
Epoch 14/50
834s - loss: 0.1113 - acc: 0.9634
Epoch 15/50
831s - loss: 0.1041 - acc: 0.9655
Epoch 16/50
829s - loss: 0.0984 - acc: 0.9670
Epoch 17/50
828s - loss: 0.0926 - acc: 0.9691
Epoch 18/50
824s - loss: 0.0870 - acc: 0.9707
Epoch 19/50
830s - loss: 0.0840 - acc: 0.9716
Epoch 20/50
827s - loss: 0.0805 - acc: 0.9726
Epoch 21/50
825s - loss: 0.0769 - acc: 0.9738
Epoch 22/50
825s - loss: 0.0749 - acc: 0.9745
Epoch 23/50
825s - loss: 0.0711 - acc: 0.9756
Epoch 24/50
827s - loss: 0.0691 - acc: 0.9763
Epoch 25/50
832s - loss: 0.0673 - acc: 0.9770
Epoch 26/50
832s - loss: 0.0654 - acc: 0.9775
Epoch 27/50
831s - loss: 0.0643 - acc: 0.9775
Epoch 28/50
832s - loss: 0.0621 - acc: 0.9786
Epoch 29/50
833s - loss: 0.0609 - acc: 0.9789
Epoch 30/50
833s - loss: 0.0593 - acc: 0.9794
Epoch 31/50
834s - loss: 0.0583 - acc: 0.9801
Epoch 32/50
833s - loss: 0.0574 - acc: 0.9802
Epoch 33/50
834s - loss: 0.0559 - acc: 0.9806
Epoch 34/50
838s - loss: 0.0550 - acc: 0.9808
Epoch 35/50
837s - loss: 0.0546 - acc: 0.9812
Epoch 36/50
838s - loss: 0.0531 - acc: 0.9817
Epoch 37/50
847s - loss: 0.0531 - acc: 0.9815
Epoch 38/50
839s - loss: 0.0517 - acc: 0.9823
Epoch 39/50
840s - loss: 0.0516 - acc: 0.9821
Epoch 40/50
839s - loss: 0.0507 - acc: 0.9824
Epoch 41/50
840s - loss: 0.0506 - acc: 0.9823
Epoch 42/50
838s - loss: 0.0498 - acc: 0.9826
Epoch 43/50
841s - loss: 0.0492 - acc: 0.9827
Epoch 44/50
843s - loss: 0.0492 - acc: 0.9827
Epoch 45/50
842s - loss: 0.0481 - acc: 0.9835
Epoch 46/50
838s - loss: 0.0478 - acc: 0.9832
Epoch 47/50
848s - loss: 0.0480 - acc: 0.9835
Epoch 48/50
850s - loss: 0.0470 - acc: 0.9838
Epoch 49/50
848s - loss: 0.0459 - acc: 0.9842
Epoch 50/50
849s - loss: 0.0467 - acc: 0.9837
50464/50476 [============================>.] - ETA: 0s  
SCORE: 1.02790703038
Model Metrics:
Accuracy 0.845372058008

    """
