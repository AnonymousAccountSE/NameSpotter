import json
import numpy as np
import pickle as pkl
import math
from tqdm import tqdm
import os

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
import re
def clean_str(string,use=True):

    if not use: return string

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def tf_idf_transform(inputs, mapping=None, sparse=False):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from scipy.sparse import coo_matrix
    vectorizer = CountVectorizer(vocabulary=mapping)

    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(inputs))
    weight = tf_idf.toarray()
    return weight if not sparse else coo_matrix(weight).toarray()
#
def PMI(inputs, mapping, window_size, sparse):
    W_ij = np.zeros([len(mapping), len(mapping)], dtype=np.float64)
    W_i = np.zeros([len(mapping)], dtype=np.float64)
    W_count = 0
    for one in inputs:
        word_list = one.split(' ')
        if len(word_list) - window_size < 0:
            window_num = 1
        else:
            window_num = len(word_list) - window_size + 1
        for i in range(window_num):
            W_count += 1
            context = list(set(word_list[i: i+window_size]))
            while '' in context:
                context.remove('')
            for j in range(len(context)):
                W_i[mapping[context[j]]] += 1
                for k in range(j + 1, len(context)):
                    W_ij[mapping[context[j]], mapping[context[k]]] += 1
                    W_ij[mapping[context[k]], mapping[context[j]]] += 1
    if sparse:
        print("sparse!!!!!")
        rows = []
        columns = []
        data = []
        for i in range(len(mapping)):
            rows.append(i)
            columns.append(i)
            data.append(1)
            tmp = [ele for ele in np.nonzero(W_ij[i])[0] if ele > i]
            for j in tmp:
                value = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j])
                if value > 0:
                    rows.append(i)
                    columns.append(j)
                    data.append(value)
                    rows.append(j)
                    columns.append(i)
                    data.append(value)
        PMI_adj = coo_matrix((data, (rows, columns)), shape=(len(mapping), len(mapping)))
    else:
        PMI_adj = np.zeros([len(mapping), len(mapping)], dtype=np.float64)
        for i in range(len(mapping)):
            PMI_adj[i, i] = 1
            tmp = [ele for ele in np.nonzero(W_ij[i])[0] if ele > i]
            for j in tmp:
                pmi = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j])
                if pmi > 0:
                    PMI_adj[i, j] = pmi
                    PMI_adj[j, i] = pmi
    return PMI_adj


def similarity_cal(word_mapping):
    sim_matrix = cosine_similarity([value[0] for key,value in word_mapping.items()])
    print("sim_matrix")
    row_num =0
    for row in sim_matrix:
        col_num = 0
        for col in row:
            if col < 0.95:
                sim_matrix[row_num][col_num] = 0
            else:
                print(sim_matrix[row_num][col_num])
            col_num = col_num + 1
        row_num = row_num + 1
    return sim_matrix

def similarity_cal_param(word_mapping, param_mapping):
    new_dict = {}
    token_set = param_mapping.keys()

    for key in token_set:
        if key == "null":
            new_dict[key] = np.zeros((1,768),dtype=np.int)
        if word_mapping.__contains__(key):
            new_dict[key] = word_mapping[key]
        else:
            new_dict[key] = np.zeros((1,768),dtype=np.int)

    sim_matrix = cosine_similarity([value[0] for key,value in new_dict.items()])
    print("sim_matrix")
    row_num =0
    for row in sim_matrix:
        col_num = 0
        for col in row:
            if col < 0.95:
                sim_matrix[row_num][col_num] = 0
            else:
                print(sim_matrix[row_num][col_num])
            col_num = col_num + 1
        row_num = row_num + 1
    print(len(sim_matrix))
    print(len(sim_matrix[0]))
    return sim_matrix


def ngram(tag_1, tag_2):
    f = open("../ngram/ngram_corpus.txt")
    lines = f.readlines()
    tag1_list = []
    tag2_list = []
    possibility_list = []
    ngram_possibility = 0
    for line in lines:
        tag1 = line.split(",")[0].strip()
        tag2 = line.split(",")[1].strip()
        possibility = line.split(",")[2]
        tag1_list.append(tag1)
        tag2_list.append(tag2)
        possibility_list.append(possibility)
    if tag_1 == tag_2:
        return 1
    for i in range(len(possibility_list)):
        if tag1_list[i].strip() == str(tag_1).upper() and tag2_list[i].strip() == str(tag_2).upper():
            ngram_possibility =  float(possibility_list[i])
            break
    return ngram_possibility


def ngram_cal(tags_mapping):
    W_ij = np.zeros([len(tags_mapping), len(tags_mapping)], dtype=np.float64)
    for i in range(len(tags_mapping)):
        tag_1 = list(tags_mapping.keys())[i]
        for j in range(len(tags_mapping)):
            tag_2 = list(tags_mapping.keys())[j]
            print(ngram(tag_1,tag_2))
            ngram_value = ngram(tag_1, tag_2)
            if ngram_value > -0.5:
                W_ij[i, j] = ngram_value
            else:
                W_ij[i, j] = 0
    print(W_ij)
    return W_ij



def construct_graph(dataset_name, remove_StopWord=False):
    os.makedirs(f'./{dataset_name}_data', exist_ok=True)
    f_train = json.load(open('./{}.json'.format(dataset_name),encoding='utf-8'))['train']
    f_test = json.load(open('./{}.json'.format(dataset_name),encoding='utf-8'))['test']

    print(len(f_train))
    print(len(f_test))
    from collections import defaultdict
    word_freq=defaultdict(int)
    param_freq=defaultdict(int)
    for item in f_train.values():
        words=clean_str(item['text']).split(' ')
        params =clean_str(item['param']).split(' ')
        for one in words:
            word_freq[one.lower()]+=1
        for one in params:
            param_freq[one.lower()]+=1
    for item in f_test.values():
        words = clean_str(item['text']).split(' ')
        params = clean_str(item['param']).split(' ')
        for one in words:
            word_freq[one.lower()]+=1
        for one in params:
            param_freq[one.lower()]+=1


    method_name_nodes = []
    tag_set = set()
    param_tag_set = set()
    param_set = set()
    words_set = set()
    train_idx = []
    test_idx = []
    labels = []
    tag_list = []
    word_list = []
    param_list = []
    #
    for i, item in enumerate(tqdm(f_train.values())):
        method_name = clean_str(item['text'])
        param = clean_str(item['param'])
        tags = item['pos'].lower().split(" ")
        if not method_name:
            print(method_name)
            continue
        if '' in tags:
            print(item)
        tag_list.append(' '.join(tags))
        tag_set.update(tags)
        labels.append(item['label'])
        words = [one.lower() for one in method_name.split(' ')]
        if '' in words:
            print(words)

        word_list.append(' '.join(words))
        words_set.update(words)

        param = [one.lower() for one in param.split(' ')]
        param_list.append(' '.join(param))
        param_set.update(param)

        if method_name:
            method_name_nodes.append(method_name)
        else:
            print(item)
            print(method_name)
        train_idx.append(len(train_idx))



    for i, item in enumerate(tqdm(f_test.values())):
        method_name = clean_str(item['text'])
        param = clean_str(item['param'])
        tags = item['pos'].lower().split(" ")
        if not method_name:
            print(method_name)
            continue

        tag_list.append(' '.join(tags))
        tag_set.update(tags)

        labels.append(item['label'])
        words = [one.lower() for one in method_name.split(' ')]
        if '' in words:
            print(words)

        word_list.append(' '.join(words))
        words_set.update(words)

        param = [one.lower() for one in param.split(' ')]
        param_list.append(' '.join(param))
        param_set.update(param)
        if method_name:
            method_name_nodes.append(method_name)
        else:
            print(item)
            print(method_name)

        test_idx.append(len(test_idx)+ len(train_idx))

    print(tag_set)
    print(param_tag_set)

    word_nodes = list(words_set)
    tag_nodes = list(tag_set)
    param_nodes = list(param_set)
    param_tag_nodes = list(param_tag_set)
    nodes_all = method_name_nodes + tag_nodes + param_nodes + word_nodes
    nodes_num = len(method_name_nodes) + len(tag_nodes) + len(param_nodes) + len(word_nodes)
    print('method_name', len(method_name_nodes))
    print('tag', len(tag_nodes))
    print('param', len(param_nodes))
    print('word', len(word_nodes))

    if len(nodes_all) != nodes_num:
        print('duplicate name error')
    print('len_train',len(train_idx))
    print('len_test',len(test_idx))
    print('len_quries',len(method_name_nodes))

    tags_mapping = {key: value for value, key in enumerate(tag_nodes)}
    words_mapping = {key: value for value, key in enumerate(word_nodes)}
    params_mapping = {key: value for value, key in enumerate(param_nodes)}
    # output tag and adjacent matrix
    adj_method_name2tag = tf_idf_transform(tag_list, tags_mapping)
    adj_tag = ngram_cal(tags_mapping)
    pkl.dump(adj_method_name2tag, open('./{}_data/adj_method_name2tag.pkl'.format(dataset_name), 'wb'))
    pkl.dump(adj_tag, open('./{}_data/adj_tag.pkl'.format(dataset_name), 'wb'))

    # output word and adjacent matrix
    # adj_method_name2word = tf_idf_transform(word_list, words_mapping, sparse=True)
    #
    # # calculate Bert vector similarity
    # str = open('../BertVector/VectorMap.json','r')
    # vector_mapping = json.load(str)
    # print("adj_word_SIM")
    # adj_word_SIM = similarity_cal(vector_mapping)
    # print(adj_word_SIM)
    # pkl.dump(adj_method_name2word, open('./{}_data/adj_method_name2word.pkl'.format(dataset_name), 'wb'))
    # pkl.dump(adj_word_SIM, open('./{}_data/adj_word.pkl'.format(dataset_name), 'wb'))

    # output param and adjacent matrix
    adj_method_name2param = tf_idf_transform(param_list, params_mapping, sparse=True)
    str = open('../BertVector/VectorParameterMap.json', 'r')
    vector_parameter_mapping = json.load(str)
    adj_param = similarity_cal_param(vector_parameter_mapping, params_mapping)
    pkl.dump(adj_method_name2param, open('./{}_data/adj_method_name2param.pkl'.format(dataset_name), 'wb'))
    pkl.dump(adj_param, open('./{}_data/adj_param.pkl'.format(dataset_name), 'wb'))

    json.dump(train_idx, open('./{}_data/train_idx.json'.format(dataset_name), 'w'), ensure_ascii=False)
    json.dump(test_idx, open('./{}_data/test_idx.json'.format(dataset_name), 'w'), ensure_ascii=False)
    json.dump(method_name_nodes, open('./{}_data/method_names.json'.format(dataset_name), 'w'), ensure_ascii=False)

    sorted_set = sorted(set(labels))
    print(sorted_set)
    label_map = {value: i for i, value in enumerate(sorted_set)}
    json.dump([label_map[label] for label in labels], open('./{}_data/labels.json'.format(dataset_name), 'w'),
              ensure_ascii=False)
    json.dump(method_name_nodes, open('./{}_data/method_name_id2_list.json'.format(dataset_name), 'w'),
              ensure_ascii=False)
    json.dump(tag_nodes, open('./{}_data/tag_id2_list.json'.format(dataset_name), 'w'), ensure_ascii=False)
    json.dump(param_nodes, open('./{}_data/param_id2_list.json'.format(dataset_name), 'w'),
              ensure_ascii=False)
    json.dump(param_tag_nodes, open('./{}_data/param_pos_id2_list.json'.format(dataset_name), 'w'),
              ensure_ascii=False)
    json.dump(word_nodes, open('./{}_data/word_id2_list.json'.format(dataset_name), 'w'), ensure_ascii=False)

dataset_name='method_name_param_pos_tag'
construct_graph(dataset_name)