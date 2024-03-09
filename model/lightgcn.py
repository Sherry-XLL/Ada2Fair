import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F

from .loss import BPRLoss
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.layers import activation_layer
from recbole.model.loss import EmbLoss
from recbole.utils import InputType


class LightGCN(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]
        self.alpha = config["alpha"]
        self.fairness_type = config["fairness_type"]
        self.fairness_weight = None

        if self.fairness_type is not None:
            self.dropout_prob = config["dropout_prob"]
            self.encoder_layers = config["encoder_layers"]
            self.encoder_activations = config["encoder_activations"]
            self.decoder_layers_acc = config["decoder_layers_acc"]
            self.decoder_activations_acc = config["decoder_activations_acc"]
            self.decoder_layers_pfair = config["decoder_layers_pfair"]
            self.decoder_activations_pfair = config["decoder_activations_pfair"]
            self.decoder_layers_ufair = config["decoder_layers_ufair"]
            self.decoder_activations_ufair = config["decoder_activations_ufair"]
            self.rating_matrix = self.get_rating_matrix(dataset)
            self.encoder = self.mlp_layers(
                [self.n_items] + self.encoder_layers + [self.n_items],
                self.encoder_activations,
                [self.dropout_prob] * (len(self.encoder_layers) + 1),
            )
            self.decoder_pfair = self.mlp_layers(
                [self.n_items] + self.decoder_layers_pfair,
                self.decoder_activations_pfair,
                [self.dropout_prob] * len(self.decoder_layers_pfair),
            )
            self.decoder_ufair = self.mlp_layers(
                [self.n_items] + self.decoder_layers_ufair,
                self.decoder_activations_ufair,
                [self.dropout_prob] * len(self.decoder_layers_ufair),
            )
            self.loss_func = torch.nn.MSELoss(reduction="sum")

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def mlp_layers(self, layer_dims, activations, dropouts):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            mlp_modules.append(activation_layer(activations[i]))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Dropout(p=dropouts[i]))
        return nn.Sequential(*mlp_modules)

    def get_rating_matrix(self, dataset):
        history_item_id, history_item_value, _ = dataset.history_item_matrix()

        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = history_item_id.flatten()
        row_indices = torch.arange(self.n_users).repeat_interleave(
            history_item_id.shape[1], dim=0
        )
        rating_matrix = torch.zeros(1).repeat(self.n_users, self.n_items)
        rating_matrix.index_put_(
            (row_indices, col_indices), history_item_value.flatten()
        )
        return rating_matrix.to(self.device)

    def get_all_weights(self):
        h_encode = self.encoder(self.rating_matrix)
        return h_encode

    def weight_loss(self, interaction):
        # Stage I: two-sided fairness aware weight generation
        user = interaction[self.USER_ID]
        provider_fairness, user_fairness = self.fairness_weight
        rating_matrix = self.rating_matrix[user]
        h_encode = self.encoder(rating_matrix)
        h_encode = h_encode * rating_matrix
        h_encode_pfair = self.decoder_pfair(h_encode)
        h_encode_ufair = self.decoder_ufair(h_encode)

        # loss for provider-side fairness
        target_pfair = self.decoder_pfair(
            (
                (provider_fairness.unsqueeze(dim=0).repeat(user.size(0), 1))
                * rating_matrix
            ).float()
        )
        loss_pfair = self.loss_func(h_encode_pfair, target_pfair)

        # loss for customer-side fairness
        target_ufair = self.decoder_ufair(
            (
                (
                    user_fairness[user]
                    .unsqueeze(dim=1)
                    .repeat(1, provider_fairness.size(0))
                )
                * rating_matrix
            ).float()
        )
        loss_ufair = self.loss_func(h_encode_ufair, target_ufair)

        return loss_pfair * (1 - self.alpha), loss_ufair * self.alpha

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        if self.fairness_weight is not None:
            user = interaction[self.USER_ID]
            pos_item = interaction[self.ITEM_ID]
            neg_item = interaction[self.NEG_ITEM_ID]
            user_all_embeddings, item_all_embeddings = self.forward()
            u_embeddings = user_all_embeddings[user]
            pos_embeddings = item_all_embeddings[pos_item]
            neg_embeddings = item_all_embeddings[neg_item]

            # calculate BPR Loss
            pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
            neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

            # weighted preference learning
            ipw = (self.fairness_weight[user, pos_item]).to(self.device)
            mf_loss = self.mf_loss(pos_scores, neg_scores, ipw)

            # calculate BPR Loss
            u_ego_embeddings = self.user_embedding(user)
            pos_ego_embeddings = self.item_embedding(pos_item)
            neg_ego_embeddings = self.item_embedding(neg_item)

            reg_loss = self.reg_loss(
                u_ego_embeddings,
                pos_ego_embeddings,
                neg_ego_embeddings,
                require_pow=self.require_pow,
            )

            loss = mf_loss + self.reg_weight * reg_loss
            return loss
        else:
            user = interaction[self.USER_ID]
            pos_item = interaction[self.ITEM_ID]
            neg_item = interaction[self.NEG_ITEM_ID]
            user_all_embeddings, item_all_embeddings = self.forward()
            u_embeddings = user_all_embeddings[user]
            pos_embeddings = item_all_embeddings[pos_item]
            neg_embeddings = item_all_embeddings[neg_item]

            # calculate BPR Loss
            pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
            neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
            mf_loss = self.mf_loss(pos_scores, neg_scores)

            # calculate BPR Loss
            u_ego_embeddings = self.user_embedding(user)
            pos_ego_embeddings = self.item_embedding(pos_item)
            neg_ego_embeddings = self.item_embedding(neg_item)

            reg_loss = self.reg_loss(
                u_ego_embeddings,
                pos_ego_embeddings,
                neg_ego_embeddings,
                require_pow=self.require_pow,
            )

            loss = mf_loss + self.reg_weight * reg_loss
            return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
