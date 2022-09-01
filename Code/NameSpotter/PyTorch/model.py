import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import aggregate
from GCN import GCN
import numpy as np


class NameSpotter(nn.Module):
    def __init__(self, adj_dict, features_dict, in_features_dim, out_features_dim, params):
        super(NameSpotter, self).__init__()
        self.threshold = params.threshold
        self.adj = adj_dict
        self.feature = features_dict
        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.type_num = len(params.type_num_node)
        self.drop_out = params.drop_out
        self.concat_word_emb = params.concat_word_emb
        self.device = params.device
        self.GCNs = []
        self.GCNs_2 = []

        # self.w1 = torch.nn.Parameter(torch.FloatTensor(50, 50), requires_grad=True)
        # self.w2 = torch.nn.Parameter(torch.FloatTensor(50, 50), requires_grad=True)
        # self.w3 = torch.nn.Parameter(torch.FloatTensor(50, 50), requires_grad=True)

        # self.w1 = torch.nn.Parameter(torch.FloatTensor(100, 100), requires_grad=True)
        # self.w2 = torch.nn.Parameter(torch.FloatTensor(100, 100), requires_grad=True)
        # self.w3 = torch.nn.Parameter(torch.FloatTensor(100, 100), requires_grad=True)

        # self.w1 = torch.nn.Parameter(torch.FloatTensor(150, 150), requires_grad=True)
        # self.w2 = torch.nn.Parameter(torch.FloatTensor(150, 150), requires_grad=True)
        # self.w3 = torch.nn.Parameter(torch.FloatTensor(150, 150), requires_grad=True)


        self.w1 = torch.nn.Parameter(torch.FloatTensor(200, 200), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(200, 200), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(200, 200), requires_grad=True)

        # self.w1 = torch.nn.Parameter(torch.FloatTensor(250, 250), requires_grad=True)
        # self.w2 = torch.nn.Parameter(torch.FloatTensor(250, 250), requires_grad=True)
        # self.w3 = torch.nn.Parameter(torch.FloatTensor(250, 250), requires_grad=True)

        # self.w1 = torch.nn.Parameter(torch.FloatTensor(300, 300), requires_grad=True)
        # self.w2 = torch.nn.Parameter(torch.FloatTensor(300, 300), requires_grad=True)
        # self.w3 = torch.nn.Parameter(torch.FloatTensor(300, 300), requires_grad=True)

        # self.w1 = torch.nn.Parameter(torch.FloatTensor(350, 350), requires_grad=True)
        # self.w2 = torch.nn.Parameter(torch.FloatTensor(350, 350), requires_grad=True)
        # self.w3 = torch.nn.Parameter(torch.FloatTensor(350, 350), requires_grad=True)

        # self.w1 = torch.nn.Parameter(torch.FloatTensor(400, 400), requires_grad=True)
        # self.w2 = torch.nn.Parameter(torch.FloatTensor(400, 400), requires_grad=True)
        # self.w3 = torch.nn.Parameter(torch.FloatTensor(400, 400), requires_grad=True)

        # torch.nn.init.normal_(self.w1)
        # torch.nn.init.normal_(self.w2)
        # torch.nn.init.normal_(self.w3)

        torch.nn.init.eye(self.w1)
        torch.nn.init.eye(self.w2)
        torch.nn.init.eye(self.w3)

        for i in range(1, self.type_num):
            self.GCNs.append(GCN(self.in_features_dim[i], self.out_features_dim[i]).to(self.device))
            self.GCNs_2.append(GCN(self.out_features_dim[i], self.out_features_dim[i]).to(self.device))

        # self.refined_linear = nn.Linear(self.out_features_dim[1]+self.out_features_dim[2]+self.out_features_dim[3]
        #                     if not self.concat_word_emb else self.in_features_dim[-1], 200)
        self.refined_linear = nn.Linear(self.out_features_dim[2], 200)
        self.final_GCN = GCN(200, self.out_features_dim[-1]).to(self.device)
        self.final_GCN_2 = GCN(self.out_features_dim[-1], self.out_features_dim[-1]).to(self.device)
        # self.FC = nn.Linear(out_features_dim[-1], out_features_dim[0])
        self.FC = nn.Linear(200, out_features_dim[0])

    # Hierarchically pooling
    def embed_component(self, norm=True):
        output = []
        for i in range(self.type_num - 1):
            # word embedding
            if i == 1 and self.concat_word_emb:
                temp_emb = torch.cat([
                    F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
                                             self.GCNs[i](self.adj[str(i + 1) + str(i + 1)], self.feature[str(i + 1)],
                                                          identity=True)),
                              p=self.drop_out, training=self.training), self.feature['word_emb']], dim=-1)
                output.append(temp_emb)
            # pos embedding
            elif i == 0:
                temp_emb = F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
                                                    self.GCNs[i](self.adj[str(i + 1) + str(i + 1)],
                                                                 self.feature[str(i + 1)], identity=True)),
                                     p=self.drop_out, training=self.training)
                output.append(temp_emb)
            # param embedding
            else:
                temp_emb = F.dropout(self.GCNs_2[i](self.adj[str(i + 1) + str(i + 1)],
                                                    self.GCNs[i](self.adj[str(i + 1) + str(i + 1)],
                                                                 self.feature[str(i + 1)], identity=True)),
                                     p=self.drop_out, training=self.training)
                output.append(temp_emb)

        # obtain the representation of word-level components.
        refined_text_input = aggregate(self.adj, output, self.type_num - 1)

        # normalize each x to unit norm
        if norm:
            refined_text_input_normed = []
            for i in range(self.type_num - 1):
                refined_text_input_normed.append(
                    refined_text_input[i] / (refined_text_input[i].norm(p=2, dim=-1, keepdim=True) + 1e-9))
        else:
            refined_text_input_normed = refined_text_input

        return refined_text_input_normed


    def forward(self, epoch):
        refined_text_input_normed = self.embed_component()
        # print(refined_text_input_normed)
        # concatenate vector along the last dimension
        # Doc_features = torch.cat(refined_text_input_normed, dim=-1)
        # print(Doc_features.size())

        # weighted average of the 3 types of subgraphs.

        weight_pos = torch.matmul(refined_text_input_normed[0], self.w1)
        # print(weight_pos.size())
        weight_word = torch.matmul(refined_text_input_normed[1], self.w2)
        weight_param = torch.matmul(refined_text_input_normed[2], self.w3)
        # print(weight_word.size())
        Doc_features = torch.add(weight_pos, weight_word)
        Doc_features = torch.add(Doc_features, weight_param)
        # Doc_features = torch.div(Doc_features, 10)
        print("Doc_features")
        print(Doc_features.shape)
        refined_text_input_after_final_linear = F.dropout(self.refined_linear(Doc_features),
                                                          p=self.drop_out, training=self.training)

        scores = self.FC(refined_text_input_after_final_linear)
        # print("scores:" + str(scores))
        return scores
    #
    # def forward(self, epoch):
    #     refined_text_input_normed = self.embed_component()
    #     # print(refined_text_input_normed)
    #     # concatenate vector along the last dimension
    #     # Doc_features = torch.cat(refined_text_input_normed, dim=-1)
    #     # print(Doc_features.size())
    #
    #     # weighted average of the 3 types of subgraphs.
    #
    #     weight_pos = torch.matmul(refined_text_input_normed[0], self.w1)
    #     # print(weight_pos.size())
    #     weight_word = torch.matmul(refined_text_input_normed[1], self.w2)
    #     weight_param = torch.matmul(refined_text_input_normed[2], self.w3)
    #     # print(weight_word.size())
    #     Doc_features = torch.add(weight_pos, weight_word)
    #     Doc_features = torch.add(Doc_features, weight_param)
    #     Doc_features = torch.div(Doc_features, 10)
    #
    #     refined_text_input_after_final_linear = F.dropout(self.refined_linear(Doc_features),
    #                                                       p=self.drop_out, training=self.training)
    #     cos_simi_total = torch.matmul(Doc_features, Doc_features.t())
    #     print(cos_simi_total.cpu().detach().numpy())
    #     # print(np.mean(cos_simi_total.cpu().detach().numpy()))
    #     # print(np.median(cos_simi_total.cpu().detach().numpy()))
    #     refined_adj_tmp = cos_simi_total * (cos_simi_total > self.threshold).float()
    #     # print(refined_adj_tmp)
    #     refined_adj = refined_adj_tmp / (refined_adj_tmp.sum(dim=-1, keepdim=True) + 1e-9)
    #     # print(refined_adj)
    #     final_text_output = self.final_GCN_2(refined_adj,
    #                                          self.final_GCN(refined_adj, refined_text_input_after_final_linear))
    #     # print(str(final_text_output))
    #     final_text_output = F.dropout(final_text_output, p=self.drop_out, training=self.training)
    #     # print(str(final_text_output))
    #     scores = self.FC(final_text_output)
    #     # print("scores:" + str(scores))
    #     return scores



