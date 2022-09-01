from collections import Counter

import numpy as np
import pickle as pkl
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import math
from sklearn import metrics
from utils import fetch_to_tensor
from model import NameSpotter
from pytorchtools import EarlyStopping
import warnings
import random
warnings.filterwarnings("ignore")
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.over_sampling import RandomUnderSampler

class Trainer(object):
    def __init__(self, params):
        self.dataset_name = params.dataset
        self.max_epoch = params.max_epoch
        self.save_path = params.save_path
        self.device = params.device
        self.hidden_size = params.hidden_size
        self.lr = params.lr
        self.weight_decay = params.weight_decay
        self.concat_word_emb = params.concat_word_emb
        self.type_names = params.type_num_node
        self.data_path = params.data_path
        self.folds = 10

        self.adj_dict, self.features_dict, self.train_idx, self.method_names_idx, self.valid_idx, self.test_idx, self.labels, self.nums_node = self.load_data()
        # self.adj_dict, self.features_dict, self.train_idx, self.valid_idx, self.test_idx, self.labels, self.nums_node = self.load_data()
        self.label_num = len(set(self.labels))
        self.labels = torch.tensor(self.labels).to(self.device)
        self.out_features_dim = [self.label_num, self.hidden_size, self.hidden_size, self.hidden_size, self.hidden_size]
        in_fea_final = self.out_features_dim[1] + self.out_features_dim[2] + self.out_features_dim[3]
        self.in_features_dim = [0, self.nums_node[1], self.nums_node[2], self.nums_node[-1], in_fea_final]

        if self.concat_word_emb:
            self.in_features_dim[-1] += self.features_dict['word_emb'].shape[-1]
        self.model = NameSpotter(self.adj_dict, self.features_dict, self.in_features_dim, self.out_features_dim, params)
        self.model = self.model.to(self.device)
        total_trainable_params = sum(p.numel() for p in self.model.parameters())
        print(self.model.state_dict())
        print(f'{total_trainable_params:,} training parameters.')
        self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim = optim.Adam([{'params': self.model.parameters()},
                                 {'params': self.model.GCNs[0].parameters()},
                                 {'params': self.model.GCNs[1].parameters()},
                                 {'params': self.model.GCNs[2].parameters()},
                                 {'params': self.model.GCNs_2[0].parameters()},
                                 {'params': self.model.GCNs_2[1].parameters()},
                                 {'params': self.model.GCNs_2[2].parameters()},], lr=self.lr, weight_decay=self.weight_decay)

    def train(self):

        global_best_acc = 0

        global_best_precision = 0
        global_best_epoch = 0
        global_evaluation = {}
        best_test_acc = 0
        best_test_f1 = 0
        best_valid_epoch = 0
        best_valid_f1=0
        best_valid_acc = 0

        acc_valid = 0
        loss_valid = 0
        f1_valid = 0
        acc_test=0
        loss_test = 0
        f1_test = 0
        best_acc = 0
        best_f1 = 0
        best_precision = 0

        average_best_acc = 0
        average_best_f1 = 0
        average_best_precision = 0
        global_best_record = ""
        all_best_evaluation = []
        for fold in range(self.folds):
            patience = 10  # 当验证集损失在连续10次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
            early_stopping = EarlyStopping(patience, verbose=True)
            print("fold" + str(fold)+"----------------")
            best_test_record = ""
            global_best_f1 = 0
            for i in range(1, self.max_epoch + 1):
                t = time.time()
                output = self.model(i)
                train_scores = output[self.train_idx[fold]]
                train_labels = self.labels[self.train_idx[fold]]
                loss = F.cross_entropy(train_scores, train_labels)
                # lf = Focal_Loss(torch.tensor([0.8,0.2]))
                # loss = lf(train_scores,train_labels)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss = loss.item()
                acc = torch.eq(torch.argmax(train_scores, dim=-1), train_labels).float().mean().item()
                print('Epoch {}  loss: {:.4f} acc: {:.4f} time{:.4f}'.format(i, loss, acc, time.time() - t))

                acc_valid, loss_valid, f1_valid, acc_test, loss_test, f1_test, precision_test, measure_result, records = self.test(i,fold)
                early_stopping(loss_valid, self.model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    # 结束模型训练
                    break

                if f1_test > global_best_f1:
                # if acc_test > global_best_acc:
                    global_best_acc = acc_test
                    global_best_f1 = f1_test
                    global_best_epoch = i
                    global_best_precision = precision_test
                    global_evaluation = measure_result
                    best_test_record = records
                if f1_valid > best_valid_f1:
                    best_valid_acc = acc_valid
                    best_valid_f1 = f1_valid
                    best_test_acc = acc_test
                    best_test_f1 = f1_test
                    best_valid_epoch = i
                best_acc = global_best_acc
                best_f1 = global_best_f1
                best_precision = global_best_precision
                best_epoch = global_best_epoch

                print('VALID: VALID ACC', best_valid_acc, ' VALID F1', best_valid_f1, 'EPOCH', best_valid_epoch)
                print('VALID: TEST ACC', best_test_acc, 'TEST F1', best_test_f1, 'EPOCH', best_valid_epoch)
                print('GLOBAL: TEST ACC', global_best_acc, 'TEST F1', global_best_f1, 'EPOCH', global_best_epoch)
                print(measure_result)
            all_best_evaluation.append(global_evaluation)
            average_best_acc = average_best_acc + global_best_acc
            average_best_f1 = average_best_f1 + global_best_f1
            average_best_precision = average_best_precision + global_best_precision
            global_best_record = global_best_record + best_test_record + "\n"

        return average_best_acc, average_best_f1, average_best_precision, all_best_evaluation, global_best_record

    def test(self, epoch,fold):
        t = time.time()
        self.model.training = False
        output = self.model(0)
        records = ""
        with torch.no_grad():
            train_scores = output[self.train_idx[fold]]
            train_labels = self.labels[self.train_idx[fold]]

            loss_train = F.cross_entropy(train_scores, train_labels).item()
            # loss = F.cross_entropy(train_scores, train_labels)
            # lf = Focal_Loss(torch.tensor([0.8,0.2]))
            # loss_train = lf(train_scores, train_labels).item()
            acc_train = torch.eq(torch.argmax(train_scores, dim=-1), train_labels).float().mean().item()
            valid_scores = output[self.valid_idx[fold]]
            valid_labels = self.labels[self.valid_idx[fold]]
            loss_valid = F.cross_entropy(valid_scores, valid_labels).item()
            # loss_valid = lf(valid_scores, valid_labels).item()
            acc_valid = torch.eq(torch.argmax(valid_scores, dim=-1), valid_labels).float().mean().item()
            f1_valid = metrics.f1_score(valid_labels.detach().cpu().numpy(), torch.argmax(valid_scores, -1).detach().cpu().numpy(), average='binary')
            print('Epoch {}  loss: {:.4f} acc: {:.4f}'.format(epoch, loss_train, acc_train),
                'Valid  loss: {:.4f}  acc: {:.4f}  f1: {:.4f}'.format(loss_valid, acc_valid, f1_valid))
            test_scores = output[self.test_idx[fold]]
            test_labels = self.labels[self.test_idx[fold]]
            loss_test = F.cross_entropy(test_scores, test_labels).item()
            acc_test = torch.eq(torch.argmax(test_scores, dim=-1), test_labels).float().mean().item()
            # print("predicted results:")
            # print(torch.argmax(test_scores, dim=-1))
            # print("test labels:")
            # print(test_labels)
            print(len(self.test_idx[fold]))
            for i,idx in enumerate(self.test_idx[fold]):
                records = records + self.method_names_idx[fold][i] + ":" + str(test_labels[i].item()) + ":" + str(torch.argmax(test_scores, dim=-1)[i].item()) + "\n"
                # print(self.method_names[idx])
                # print(torch.argmax(test_scores, dim=-1)[i].item())
                # print(test_labels[i].item())
            f1_test = metrics.f1_score(test_labels.detach().cpu().numpy(), torch.argmax(test_scores,-1).detach().cpu().numpy(), average='binary')
            precision_test = metrics.precision_score(test_labels.detach().cpu().numpy(), torch.argmax(test_scores,-1).detach().cpu().numpy())
            print('Test  loss: {:.4f} acc: {:.4f} f1: {:.4f} time: {:.4f}'.format(loss_test, acc_test, f1_test, time.time() - t))
            measure_result = metrics.classification_report(test_labels.detach().cpu().numpy(), torch.argmax(test_scores,-1).detach().cpu().numpy(),output_dict=True)
            measure_result1 = metrics.classification_report(test_labels.detach().cpu().numpy(),
                                                            torch.argmax(test_scores, -1).detach().cpu().numpy())
            # print(precision_test)
            # print(measure_result1)
        self.model.training = True
        return acc_valid, loss_valid, f1_valid, acc_test, loss_test, f1_test,precision_test, measure_result, records

    def load_data(self):
        start = time.time()
        adj_dict = {}
        feature_dict = {}
        nums_node = []
        for i in range(1, len(self.type_names)):
            adj_dict[str(0) + str(i)] = pkl.load(
                open(self.data_path + './adj_{}2{}.pkl'.format(self.type_names[0], self.type_names[i]), 'rb'))
            if i == 1:
                nums_node.append(adj_dict[str(0) + str(i)].shape[0])

            adj_dict[str(i) + str(i)] = pkl.load(
                open(self.data_path + './adj_{}.pkl'.format(self.type_names[i]), 'rb'))
            nums_node.append(adj_dict[str(i) + str(i)].shape[0])

            feature_dict[str(i)] = np.eye(nums_node[i], dtype=np.float64)

        # feature_dict['word_emb'] = torch.tensor(pkl.load(
        #     open(self.data_path + './word_emb.pkl', 'rb')), dtype=torch.float).to(self.device)

        # adj_dict['22'] = np.array(adj_dict['22'].toarray())
        adj_dict['22'] = np.array(adj_dict['22'])
        adj_dict['33'] = np.array(adj_dict['33'])
        adj_dict['02'] = np.array(adj_dict['02'])
        adj_dict['03'] = np.array(adj_dict['03'])

        adj = {}
        feature = {}
        for i in adj_dict.keys():
            adj[i] = fetch_to_tensor(adj_dict, i, self.device)
        for i in feature_dict.keys():
            feature[i] = fetch_to_tensor(feature_dict, i, self.device)

        labels = json.load(open(self.data_path + './labels.json'))
        method_names = json.load(open(self.data_path + './method_names.json'))
        label_dict = {}
        train_idx = []  # 创建一个空列表
        for i in range(10):  # 创建一个5行的列表（行）
            train_idx.append([])  # 在空的列表中添加空的列表

        valid_idx = []  # 创建一个空列表
        for i in range(10):  # 创建一个5行的列表（行）
            valid_idx.append([])  # 在空的列表中添加空的列表

        test_idx = []  # 创建一个空列表
        for i in range(10):  # 创建一个5行的列表（行）
            test_idx.append([])  # 在空的列表中添加空的列表

        method_names_idx = []  # 创建一个空列表
        for i in range(10):  # 创建一个5行的列表（行）
            method_names_idx.append([])  # 在空的列表中添加空的列表

        for i in set(labels):
            label_dict[i] = []
        for j, label in enumerate(labels):
            label_dict[label].append(j)
        # len_train_idx = len(label_dict) * 253

        # each_fold_num = int(len(labels)/20)
        # print("each_fold_num:" + str(each_fold_num))

        negative_fold_num = int(len(label_dict[0]) / 10)
        positive_fold_num = int(len(label_dict[1]) / 10)

        # for fold in range(10):
        #     print(fold)
        #     for i, idxes in enumerate(label_dict.values()):
        #         np.random.shuffle(idxes)
        #         # print("len(idxs):"+str(len(idxs)))
        #         idxs = idxes[0:len(idxes)-2]
        #         if fold>=3:
        #             train_idx[fold].extend(idxs[0: each_fold_num*fold + 8*each_fold_num - len(idxs)] + idxs[each_fold_num*fold + 10*each_fold_num - len(idxs):])
        #             valid_idx[fold].extend(idxs[each_fold_num*fold + 9*each_fold_num - len(idxs): each_fold_num*fold + 10*each_fold_num - len(idxs)])
        #             test_idx[fold].extend(idxs[each_fold_num*fold + 10*each_fold_num - len(idxs): each_fold_num*fold + 11*each_fold_num - len(idxs)])
        #         elif fold == 1:
        #             train_idx[fold].extend(idxs[each_fold_num * fold: each_fold_num * fold + 8 * each_fold_num])
        #             valid_idx[fold].extend(idxs[each_fold_num * fold + 8 * each_fold_num : each_fold_num * fold + 9 * each_fold_num])
        #             test_idx[fold].extend(idxs[0:each_fold_num * fold])
        #         elif fold == 2:
        #             train_idx[fold].extend(idxs[each_fold_num * fold: each_fold_num * fold + 8 * each_fold_num])
        #             valid_idx[fold].extend(idxs[0: each_fold_num])
        #             test_idx[fold].extend(idxs[each_fold_num: each_fold_num + each_fold_num])
        #         else:
        #             train_idx[fold].extend(idxs[each_fold_num * fold: each_fold_num * fold + 8 * each_fold_num])
        #             valid_idx[fold].extend(idxs[each_fold_num * fold + 8 * each_fold_num: each_fold_num * fold + 9 * each_fold_num])
        #             test_idx[fold].extend(idxs[each_fold_num * fold + 9 * each_fold_num: each_fold_num * fold + 10 * each_fold_num])
        #     print(len(train_idx[fold]))
        #     print(len(valid_idx[fold]))
        #     print(len(test_idx[fold]))

        for fold in range(10):
            print(fold)
            for i, idxes in enumerate(label_dict.values()):
                if i == 1:
                    # np.random.shuffle(idxes)
                    # print("len(idxs):"+str(len(idxs)))
                    idxs = idxes[0:len(idxes) - 2]
                    if fold >= 3:
                        train_idx[fold].extend(idxs[0: positive_fold_num * fold + 8 * positive_fold_num - len(idxs)] + idxs[positive_fold_num * fold + 10 * positive_fold_num - len(idxs):])
                        valid_idx[fold].extend(idxs[positive_fold_num * fold + 8 * positive_fold_num - len(idxs): positive_fold_num * fold + 9 * positive_fold_num - len(idxs)])
                        test_idx[fold].extend(idxs[positive_fold_num * fold + 9 * positive_fold_num - len(idxs): positive_fold_num * fold + 10 * positive_fold_num - len(idxs)])
                    elif fold == 1:
                        train_idx[fold].extend(
                            idxs[positive_fold_num * fold: positive_fold_num * fold + 8 * positive_fold_num])
                        valid_idx[fold].extend(
                            idxs[
                            positive_fold_num * fold + 8 * positive_fold_num: positive_fold_num * fold + 9 * positive_fold_num])
                        test_idx[fold].extend(idxs[0:positive_fold_num * fold])
                    elif fold == 2:
                        train_idx[fold].extend(
                            idxs[positive_fold_num * fold: positive_fold_num * fold + 8 * positive_fold_num])
                        valid_idx[fold].extend(idxs[0: positive_fold_num])
                        test_idx[fold].extend(idxs[positive_fold_num: positive_fold_num + positive_fold_num])
                    else:
                        train_idx[fold].extend(
                            idxs[positive_fold_num * fold: positive_fold_num * fold + 8 * positive_fold_num])
                        valid_idx[fold].extend(
                            idxs[
                            positive_fold_num * fold + 8 * positive_fold_num: positive_fold_num * fold + 9 * positive_fold_num])
                        test_idx[fold].extend(
                            idxs[
                            positive_fold_num * fold + 9 * positive_fold_num: positive_fold_num * fold + 10 * positive_fold_num])
                if i == 0:
                    # np.random.shuffle(idxes)
                    # print("len(idxs):"+str(len(idxs)))
                    idxs = idxes[0:len(idxes) - 2]
                    if fold >= 3:
                        test_idx[fold].extend(idxs[negative_fold_num * fold + 9 * negative_fold_num - len(
                            idxs): negative_fold_num * fold + 10 * negative_fold_num - len(idxs)])

                        temp_train = [item for item in idxs if item not in test_idx[fold]]
                        train_idx[fold].extend(random.sample(temp_train, positive_fold_num * 8))

                        temp_valid = [item for item in temp_train if item not in train_idx[fold]]
                        valid_idx[fold].extend(random.sample(temp_valid, positive_fold_num))
                    elif fold == 1:
                        test_idx[fold].extend(idxs[0:negative_fold_num * fold])

                        temp_train = [item for item in idxs if item not in test_idx[fold]]
                        train_idx[fold].extend(random.sample(temp_train, positive_fold_num * 8))

                        temp_valid = [item for item in temp_train if item not in train_idx[fold]]
                        valid_idx[fold].extend(random.sample(temp_valid, positive_fold_num))

                    elif fold == 2:
                        test_idx[fold].extend(idxs[negative_fold_num: negative_fold_num + negative_fold_num])

                        temp_train = [item for item in idxs if item not in test_idx[fold]]
                        train_idx[fold].extend(random.sample(temp_train, positive_fold_num * 8))

                        temp_valid = [item for item in temp_train if item not in train_idx[fold]]
                        valid_idx[fold].extend(random.sample(temp_valid, positive_fold_num))
                    else:
                        test_idx[fold].extend(
                            idxs[
                            negative_fold_num * fold + 9 * negative_fold_num: negative_fold_num * fold + 10 * negative_fold_num])

                        temp_train = [item for item in idxs if item not in test_idx[fold]]
                        train_idx[fold].extend(random.sample(temp_train, positive_fold_num * 8))

                        temp_valid = [item for item in temp_train if item not in train_idx[fold]]
                        valid_idx[fold].extend(random.sample(temp_valid, positive_fold_num))

            print(len(train_idx[fold]))
            print(len(valid_idx[fold]))
            print(len(test_idx[fold]))
            print(test_idx[fold])
            for i in test_idx[fold]:
                method_names_idx[fold].append(method_names[i])
            print(len(method_names_idx[fold]))
        print(len(train_idx))
        print(len(valid_idx))
        print(len(test_idx))
        print('data process time: {}'.format(time.time() - start))

        return adj, feature, train_idx, method_names_idx, valid_idx, test_idx, labels, nums_node
        # return adj, feature, train_idx, valid_idx, test_idx, labels, nums_node

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.model.state_dict(), path + './{}/save_model_new'.format(self.dataset_name))

    def load(self):
        self.model.load_state_dict(torch.load(self.save_path))
