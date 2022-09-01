import json
import numpy as np
import pickle as pkl
import math
import nltk
from tqdm import tqdm
import os

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix
import re
def clean_str(string,use=True):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
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

def load_stopwords(filepath='./stopwords_en.txt'):
    stopwords = set()
    with open(filepath, 'r') as f:
        for line in f:
            swd = line.strip()
            stopwords.add(swd)
    print(len(stopwords))
    return stopwords

def tf_idf_transform(inputs, mapping=None, sparse=False):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from scipy.sparse import coo_matrix
    vectorizer = CountVectorizer(vocabulary=mapping)

    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(inputs))
    weight = tf_idf.toarray()

    # X = vectorizer.fit_transform(inputs)
    # weight = X.toarray()

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
            # for j in range(i + 1, len(mapping)):
            for j in tmp:
                pmi = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j])
                if pmi > 0:
                    PMI_adj[i, j] = pmi
                    PMI_adj[j, i] = pmi
    # print(PMI_adj)
    return PMI_adj


def similarity_cal(word_mapping):
    sim_matrix = cosine_similarity([value[0] for key,value in word_mapping.items()])
    print("sim_matrix")
    # print(sim_matrix)
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
    # print(sim_matrix)
    # print(np.mean(sim_matrix))
    # print(np.median(sim_matrix))
    # print(len(sim_matrix))
    # print(len(sim_matrix[0]))
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
    # print(sim_matrix)
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
    # print(sim_matrix)
    # print(np.mean(sim_matrix))
    # print(np.median(sim_matrix))
    print(len(sim_matrix))
    print(len(sim_matrix[0]))
    return sim_matrix


def ngram(tag_1, tag_2):
    # f = open("../ngram/ngram.txt")
    f = open("../ngram/ngram_corpus.txt")
    lines = f.readlines()
    tag1_list = []
    tag2_list = []
    possibility_list = []
    ngram_possibility = 0
    for line in lines:
        tag1 = line.split(",")[0].strip()
        tag2 = line.split(",")[1].strip()
        # print(tag1+" " + tag2)
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



def make_node2id_eng_text(dataset_name, remove_StopWord=False):
    stop_word=load_stopwords()
    stop_word.add('')
    os.makedirs(f'./{dataset_name}_data', exist_ok=True)

    # f_train = json.load(open('./{}_split.json'.format(dataset_name),encoding='utf-8'))['train']
    # f_test = json.load(open('./{}_split.json'.format(dataset_name),encoding='utf-8'))['test']
    # f_train = json.load(open('./{}_split_binary.json'.format(dataset_name),encoding='utf-8'))['train']
    # f_test = json.load(open('./{}_split_binary.json'.format(dataset_name),encoding='utf-8'))['test']
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


    query_nodes = []
    tag_set = set()
    param_tag_set = set()
    param_set = set()
    words_set = set()
    train_idx = []
    test_idx = []
    labels = []
    tag_list = []
    param_tag_list = []
    word_list = []
    param_list = []
    #
    for i, item in enumerate(tqdm(f_train.values())):
        # item=f_train[str(i)]
        query = clean_str(item['text'])
        param = clean_str(item['param'])
        tags = item['pos'].lower().split(" ")
        # param_tags = item['param_pos'].lower().split(" ")
        # print("tags")
        # print(tags)
        # print(query)
        if not query:
            print(query)
            continue
        # tags = [one[1].lower() for one in nltk.pos_tag(nltk.word_tokenize(query))]
        # # print(tags)
        if '' in tags:
            print(item)

        tag_list.append(' '.join(tags))
        # param_tag_list.append(' '.join(param_tags))
        # print("tag_list")
        # print(tag_list)
        tag_set.update(tags)
        # param_tag_set.update(param_tags)
        labels.append(item['label'])
        if remove_StopWord:
            words = [one.lower() for one in query.split(' ') if one not in stop_word]
        else:
            words = [one.lower() for one in query.split(' ')]
        if '' in words:
            print(words)

        word_list.append(' '.join(words))
        words_set.update(words)

        param = [one.lower() for one in param.split(' ')]
        param_list.append(' '.join(param))
        param_set.update(param)

        if query:
            query_nodes.append(query)
        else:
            print(item)
            print(query)
        train_idx.append(len(train_idx))



    for i, item in enumerate(tqdm(f_test.values())):
        # item = f_test[str(i)]
        query = clean_str(item['text'])
        param = clean_str(item['param'])
        tags = item['pos'].lower().split(" ")
        # param_tags = item['param_pos'].lower().split(" ")
        # print(query)
        if not query:
            print(query)
            continue
        # tags = [one[1].lower() for one in nltk.pos_tag(nltk.word_tokenize(query))]

        tag_list.append(' '.join(tags))
        tag_set.update(tags)
        # param_tag_list.append(' '.join(param_tags))
        # param_tag_set.update(param_tags)
        labels.append(item['label'])
        if remove_StopWord:
            words = [one.lower() for one in query.split(' ') if one not in stop_word]
        else:
            words = [one.lower() for one in query.split(' ')]
        if '' in words:
            print(words)

        word_list.append(' '.join(words))
        words_set.update(words)

        param = [one.lower() for one in param.split(' ')]
        param_list.append(' '.join(param))
        param_set.update(param)
        if query:
            query_nodes.append(query)
        else:
            print(item)
            print(query)

        test_idx.append(len(test_idx)+ len(train_idx))

    print(tag_set)
    print(param_tag_set)

    word_nodes = sorted(list(words_set))
    tag_nodes = sorted(list(tag_set))
    param_nodes = sorted(list(param_set))
    param_tag_nodes = sorted(list(param_tag_set))
    # nodes_all = list(query_nodes | tag_nodes | entity_nodes)
    nodes_all = query_nodes + tag_nodes + param_nodes + word_nodes + param_tag_nodes
    nodes_num = len(query_nodes) + len(tag_nodes) + len(param_nodes) + len(word_nodes) + len(param_tag_nodes)
    print('query', len(query_nodes))
    print('tag', len(tag_nodes))
    print('param', len(param_nodes))
    print('word', len(word_nodes))
    print('param_tag', len(param_tag_nodes))

    if len(nodes_all) != nodes_num:
        print('duplicate name error')

    print('len_train',len(train_idx))
    print('len_test',len(test_idx))
    print('len_quries',len(query_nodes))

    tags_mapping = {key: value for value, key in enumerate(tag_nodes)}
    words_mapping = {key: value for value, key in enumerate(word_nodes)}
    params_mapping = {key: value for value, key in enumerate(param_nodes)}
    params_tag_mapping = {key: value for value, key in enumerate(param_tag_nodes)}
    # output tag and adjacent matrix
    adj_query2tag = tf_idf_transform(tag_list, tags_mapping)
    # adj_tag = PMI(tag_list, tags_mapping, window_size=5, sparse=False)
    adj_tag = ngram_cal(tags_mapping)
    pkl.dump(adj_query2tag, open('./{}_data/adj_query2tag.pkl'.format(dataset_name), 'wb'))
    pkl.dump(adj_tag, open('./{}_data/adj_tag.pkl'.format(dataset_name), 'wb'))

    # output word and adjacent matrix
    adj_query2word = tf_idf_transform(word_list, words_mapping, sparse=True)
    # adj_word_PMI = PMI(word_list, words_mapping, window_size=5, sparse=True)
    # print("adj_word_PMI:")
    # print(adj_word_PMI.toarray())
    # calculate Bert vector similarity
    str = open('../BertVector/VectorMap.json','r')
    vector_mapping = json.load(str)
    print("adj_word_SIM")
    adj_word_SIM = similarity_cal(vector_mapping)
    print(adj_word_SIM)
    # adj_word = np.matmul(adj_word_SIM,adj_word_PMI.toarray())
    # print("adj_word")
    # print(adj_word)
    pkl.dump(adj_query2word, open('./{}_data/adj_query2word.pkl'.format(dataset_name), 'wb'))
    # pkl.dump(adj_word, open('./{}_data/adj_word.pkl'.format(dataset_name), 'wb'))
    # pkl.dump(adj_word_PMI, open('./{}_data/adj_word.pkl'.format(dataset_name), 'wb'))
    pkl.dump(adj_word_SIM, open('./{}_data/adj_word.pkl'.format(dataset_name), 'wb'))

    # output param and adjacent matrix
    adj_query2param = tf_idf_transform(param_list, params_mapping, sparse=True)
    # adj_param = PMI(param_list, params_mapping, window_size=5, sparse=True)
    str = open('../BertVector/VectorParameterMap.json', 'r')
    vector_parameter_mapping = json.load(str)
    print("adj_param_SIM")
    adj_param = similarity_cal_param(vector_parameter_mapping, params_mapping)
    print(adj_word_SIM)
    pkl.dump(adj_query2param, open('./{}_data/adj_query2param.pkl'.format(dataset_name), 'wb'))
    pkl.dump(adj_param, open('./{}_data/adj_param.pkl'.format(dataset_name), 'wb'))

    # # output param pos and adjacent matrix
    # adj_query2param_pos = tf_idf_transform(param_tag_list, params_tag_mapping, sparse=True)
    # # adj_param_pos = PMI(param_tag_list, params_tag_mapping, window_size=5, sparse=True)
    # adj_param_pos = ngram_cal(params_tag_mapping)
    # pkl.dump(adj_query2param_pos, open('./{}_data/adj_query2param_pos.pkl'.format(dataset_name), 'wb'))
    # pkl.dump(adj_param_pos, open('./{}_data/adj_param_pos.pkl'.format(dataset_name), 'wb'))

    json.dump(train_idx, open('./{}_data/train_idx.json'.format(dataset_name), 'w'), ensure_ascii=False)
    json.dump(test_idx, open('./{}_data/test_idx.json'.format(dataset_name), 'w'), ensure_ascii=False)
    json.dump(query_nodes, open('./{}_data/method_names.json'.format(dataset_name), 'w'), ensure_ascii=False)
    #
    sorted_set = sorted(set(labels))
    print(sorted_set)
    label_map = {value: i for i, value in enumerate(sorted_set)}
    json.dump([label_map[label] for label in labels], open('./{}_data/labels.json'.format(dataset_name), 'w'),
              ensure_ascii=False)
    json.dump(query_nodes, open('./{}_data/query_id2_list.json'.format(dataset_name), 'w'),
              ensure_ascii=False)
    json.dump(tag_nodes, open('./{}_data/tag_id2_list.json'.format(dataset_name), 'w'), ensure_ascii=False)
    json.dump(param_nodes, open('./{}_data/param_id2_list.json'.format(dataset_name), 'w'),
              ensure_ascii=False)
    json.dump(param_tag_nodes, open('./{}_data/param_pos_id2_list.json'.format(dataset_name), 'w'),
              ensure_ascii=False)
    json.dump(word_nodes, open('./{}_data/word_id2_list.json'.format(dataset_name), 'w'), ensure_ascii=False)

    # glove_emb = pkl.load(open('../data/old_glove_6B/embedding_glove.p', 'rb'))
    # vocab = pkl.load(open('../data/old_glove_6B/vocab.pkl', 'rb'))
    # embs = []
    # err_count = 0
    # for word in word_nodes:
    #     if word in vocab:
    #         embs.append(glove_emb[vocab[word]])
    #     else:
    #         err_count += 1
    #         # print('error:', word)
    #         embs.append(np.zeros(300, dtype=np.float64))
    # print('err in word count', err_count)
    # pkl.dump(np.array(embs, dtype=np.float64), open('./{}_data/word_emb.pkl'.format(dataset_name), 'wb'))

# dataset_name='method_name'
# dataset_name='method_name_param_pos'
# dataset_name='method_name_param_pos'
dataset_name='method_name_param_pos_tag_4504'
# dataset_name='snippets'
# if dataset_name in ['mr', 'snippets', 'tagmynews']:
#     remove_StopWord = True
# else:
#     remove_StopWord = False
# make_node2id_eng_text(dataset_name, remove_StopWord)
make_node2id_eng_text(dataset_name)