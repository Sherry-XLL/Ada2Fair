r"""
BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import Counter
from .loss import BPRLoss

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import activation_layer
from recbole.utils import InputType


class BPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPR, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.fairness_type = config["fairness_type"]
        self.fairness_weight = None

        if self.fairness_type is not None:
            self.alpha = config["alpha"]
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
            self.loss_func = nn.MSELoss(reduction="sum")

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self.loss = BPRLoss()
        self.other_parameter()

        # parameters initialization
        self.apply(xavier_normal_initialization)

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
        target_pfair = self.decoder_pfair(
            (
                (provider_fairness.unsqueeze(dim=0).repeat(user.size(0), 1))
                * rating_matrix
            ).float()
        )
        target_ufair = self.decoder_ufair((user_fairness[user] * rating_matrix).float())
        loss_pfair = self.loss_func(h_encode_pfair, target_pfair)
        loss_ufair = self.loss_func(h_encode_ufair, target_ufair)
        return loss_pfair * (1 - self.alpha), loss_ufair * self.alpha

    def get_user_embedding(self, user):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        if self.fairness_weight is not None:
            # Stage II: weighted preference learning
            user = interaction[self.USER_ID]
            pos_item = interaction[self.ITEM_ID]
            neg_item = interaction[self.NEG_ITEM_ID]
            user_e, pos_e = self.forward(user, pos_item)
            neg_e = self.get_item_embedding(neg_item)
            pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(
                dim=1
            ), torch.mul(user_e, neg_e).sum(dim=1)
            ipw = (self.fairness_weight[user, pos_item]).to(self.device)
            loss = self.loss(pos_item_score, neg_item_score, ipw)
            return loss
        else:
            user = interaction[self.USER_ID]
            pos_item = interaction[self.ITEM_ID]
            neg_item = interaction[self.NEG_ITEM_ID]
            user_e, pos_e = self.forward(user, pos_item)
            neg_e = self.get_item_embedding(neg_item)
            pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(
                dim=1
            ), torch.mul(user_e, neg_e).sum(dim=1)
            loss = self.loss(pos_item_score, neg_item_score)
            return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
