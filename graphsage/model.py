import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from sklearn.metrics import roc_curve, confusion_matrix, recall_score, f1_score, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import pandas as pd
import numpy as np
import random
import torch
import time
import matplotlib.pyplot as plt

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator


def performance(y_true, y_pred, name="none", write_flag=False, print_flag=False):
    f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    output = "F1-Score     %0.4f\n" %  (f1_score(y_true, y_pred))

    if write_flag:
        f = open("./results_{0}.txt".format(name), "w")
        f.write(output)
        f.close()

    if print_flag:
        print(output, end="")
        print(confusion_matrix(y_true, y_pred))


class SupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, enc, w):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.w = w
        self.xent = nn.CrossEntropyLoss(weight=self.w)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def load_suspended():
    num_nodes = 100386
    num_feats = 300
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("suspended/users_suspended.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("hate/users.edges") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    # print(label_map)
    return feat_data, labels, adj_lists


def run_suspended():
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    num_nodes = 100386
    feat_data, labels, adj_lists = load_suspended()
    features = nn.Embedding(num_nodes, 300)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 300, 256, adj_lists, agg1, gcn=False, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 256, adj_lists, agg2,
                   base_model=enc1, gcn=False, cuda=False)
    enc1.num_samples = 25
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(2, enc2, torch.FloatTensor([1, 10]))

    test = np.load("suspended/test_indexes_sa.npy")
    val = np.load("suspended/val_indexes_sa.npy")
    train = np.load("suspended/train_indexes_sa.npy")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, graphsage.parameters()), weight_decay=0.1)
    times = []
    cum_loss = 0

    for batch in range(500):
        batch_nodes = train[:128]
        train = np.roll(train, 128)
        # random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        cum_loss += loss.data[0]
        if batch % 100 == 0:
            print(batch, cum_loss / 30)
            cum_loss = 0

            val_output = graphsage.forward(val)
            y_pred = val_output.data.numpy().argmax(axis=1)
            y_true = labels[val].flatten()

            performance(y_true, y_pred, print_flag=True)
            print("Average batch time:", np.mean(times))

    val_output = graphsage.forward(test)
    y_pred = val_output.data.numpy().argmax(axis=1)
    y_true = labels[test].flatten()

    performance(y_true, y_pred, print_flag=True, name="Test")


def load_hate(features, edges, num_features):
    num_nodes = 100386
    num_feats = num_features
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}

    with open(features) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open(edges) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    print(label_map)
    return feat_data, labels, adj_lists


def run_hate(gcn, features, edges, num_features=320):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    num_nodes = 100386
    feat_data, labels, adj_lists = load_hate(features ,edges, num_features)
    features = nn.Embedding(num_nodes, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, num_features, 256, adj_lists, agg1, gcn=gcn, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 256, adj_lists, agg2,
                   base_model=enc1, gcn=gcn, cuda=False)
    enc1.num_samples = 25
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(3, enc2, torch.FloatTensor([1, 0, 10]))

    df = pd.read_csv("hate/users_anon.csv")
    df = df[df.hate != "other"]
    y = np.array([1 if v == "hateful" else 0 for v in df["hate"].values])
    x = np.array(df["user_id"].values)
    del df

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    recall_test = []
    accuracy_test = []
    auc_test = []
    for train_index, test_index in skf.split(x, y):
        train, test = x[train_index], x[test_index]

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, graphsage.parameters()))
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.01)
        times = []
        cum_loss = 0

        for batch in range(1000):
            batch_nodes = train[:128]
            train = np.roll(train, 128)
            # random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time - start_time)
            cum_loss += loss.data[0]
            if batch % 50 == 0:
                val_output = graphsage.forward(test)
                labels_pred_validation = val_output.data.numpy().argmax(axis=1)
                labels_true_validation = labels[test].flatten()
                y_true = [1 if v == 2 else 0 for v in labels_true_validation]
                y_pred = [1 if v == 2 else 0 for v in labels_pred_validation]
                fscore = f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
                # print(batch, cum_loss / 30, fscore)
                cum_loss = 0

                if fscore > 0.65:
                    break


        val_output = graphsage.forward(test)
        labels_pred_score = val_output.data.numpy()[:, 2].flatten() - val_output.data.numpy()[:, 0].flatten()
        labels_true_test = labels[test].flatten()

        y_true = [1 if v == 2 else 0 for v in labels_true_test]

        fpr, tpr, _ = roc_curve(y_true, labels_pred_score)

        labels_pred_test = labels_pred_score > 0
        auc_test.append(auc(fpr, tpr))
        y_pred = [1 if v else 0 for v in labels_pred_test]

        accuracy_test.append(accuracy_score(y_true, y_pred))
        recall_test.append(recall_score(y_true, y_pred, pos_label=1))

        # print(confusion_matrix(y_true, y_pred))
        # print("Precision   %0.4f" % accuracy_test[-1])
        # print("Recall   %0.4f" % recall_test[-1])
        # print("AUC   %0.4f" % auc_test[-1])
        # plt.figure()
        # lw = 2
        # plt.plot(fpr, tpr, color='darkorange',
        #          lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()

    accuracy_test = np.array(accuracy_test)
    recall_test = np.array(recall_test)
    auc_test = np.array(auc_test)

    print("Accuracy   %0.4f +-  %0.4f" % (accuracy_test.mean(), accuracy_test.std()))
    print("Recall    %0.4f +-  %0.4f" % (recall_test.mean(), recall_test.std()))
    print("AUC    %0.4f +-  %0.4f" % (auc_test.mean(), auc_test.std()))




if __name__ == "__main__":
    print("GraphSage all hate")
    run_hate(gcn=False, edges="hate/users.edges",
             features="hate/users_hate_all.content",
             num_features=320)
    print("GraphSage glove hate")
    run_hate(gcn=False,
             edges="hate/users.edges",
             features="hate/users_hate_glove.content",
             num_features=300)
    print("GCN all hate")
    run_hate(gcn=True, edges="hate/users.edges",
             features="hate/users_hate_all.content",
             num_features=320)
    print("GCN glove hate")
    run_hate(gcn=True,
             edges="hate/users.edges",
             features="hate/users_hate_glove.content",
             num_features=300)


#Accuracy   0.8638 +-  0.0146
# Recall    0.8254 +-  0.0221
# AUC    0.9132 +-  0.0141
