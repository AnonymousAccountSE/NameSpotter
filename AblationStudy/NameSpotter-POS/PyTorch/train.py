from __future__ import division
from __future__ import print_function
import argparse
from utils import save_res, set_seed, save_res1
from Trainer import Trainer
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "model params")
    parser.add_argument("--gpu", type=int, default=0,
                            help="choose which GPU")
    parser.add_argument("--dataset", "-d", type=str, default='method_name_param_pos_tag',
                            help="set the dataset name")
    parser.add_argument("--data_path", "-d_path", type=str, default='../preprocess/',
                            help="choose the data path if necessary")
    parser.add_argument("--save_path", type=str, default="../result/",
                            help="save path")
    parser.add_argument('--disable_cuda', action='store_true',
                            help='disable CUDA')
    parser.add_argument("--seed", type=int, default=100,
                            help="seeds for random initial")
    parser.add_argument("--hidden_size", type=int, default=200,
                            help="hidden size")
    parser.add_argument("--lr", type=float, default=1e-3,
                            help="learning rate of the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                            help="adjust the learning rate via epochs")
    parser.add_argument("--drop_out", type=float, default=0.1,
                            help="dropout rate")
    parser.add_argument("--max_epoch", type=int, default=10000,
                            help="max numer of epochs")
    parser.add_argument("--concat_word_emb", type=bool, default=False,
                            help="concat word embedding with pretrained model")
    params = parser.parse_args()
    params.type_num_node = ['method_name', 'word','param']
    params.data_path = params.data_path + './{}_data/'.format(params.dataset)
    params.save_name = params.save_path + './result_torch_{}.json'.format(params.dataset)
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')
    set_seed(params.seed)
    trainer = Trainer(params)
    test_acc,best_f1, best_precision, global_evaluation, records = trainer.train()
    print(records)
    print("best_acc:" + str(test_acc/10) + "," + "best_f1:" + str(best_f1/10) + ",best_precision:" + str(best_precision/10))
    average_precision_0 = 0
    average_recall_0 = 0
    average_f1_0 = 0
    average_precision_1 = 0
    average_recall_1 = 0
    average_f1_1 = 0

    for result in global_evaluation:
        result_0 = result["0"]
        result_1 = result["1"]
        average_precision_0 = average_precision_0 + result_0["precision"]
        average_recall_0 = average_recall_0 + result_0["recall"]
        average_f1_0 = average_f1_0 + result_0["f1-score"]
        average_precision_1 = average_precision_1 + result_1["precision"]
        average_recall_1 = average_recall_1 + result_1["recall"]
        average_f1_1 = average_f1_1 + result_1["f1-score"]

    average_precision_0 = average_precision_0 / 10
    average_recall_0 = average_recall_0 / 10
    average_f1_0 = average_f1_0 / 10
    average_precision_1 = average_precision_1/10
    average_recall_1 = average_recall_1/10
    average_f1_1 = average_f1_1/10

    print("0:" + str(round(average_precision_0,3))+","+str(round(average_recall_0,3))+"," + str(round(average_f1_0,3)))
    print("1:" + str(round(average_precision_1,3))+","+str(round(average_recall_1,3))+"," + str(round(average_f1_1,3)))
    save_res1(params, test_acc/10, best_f1/10)
    del trainer