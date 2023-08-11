import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from logging import getLogger
from time import time
from evaluator import Evaluator, Collector
from recbole.trainer import Trainer as RecBole_Trainer
from recbole.data.dataloader import UserDataLoader
from recbole.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    dict2str,
    get_tensorboard,
    set_color,
    WandbLogger,
)


class Trainer(RecBole_Trainer):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.wandblogger = WandbLogger(config)
        self.learner = config["learner"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.stopping_step = config["stopping_step"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.valid_metric = config["valid_metric"].lower()
        self.valid_metric_bigger = config["valid_metric_bigger"]
        self.test_batch_size = config["eval_batch_size"]
        self.gpu_available = torch.cuda.is_available() and config["use_gpu"]
        self.device = config["device"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.enable_amp = config["enable_amp"]
        self.enable_scaler = torch.cuda.is_available() and config["enable_scaler"]
        ensure_dir(self.checkpoint_dir)
        saved_model_file = "{}-{}.pth".format(self.config["model"], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config["weight_decay"]

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.eval_type = config["eval_type"]
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_tensor = None
        self.tot_item_num = None

    def train_weight_epoch(
        self, weight_epoch_idx=0, weight_data=None, show_progress=False
    ):
        self.model.train()
        total_loss = None
        iter_data = (
            tqdm(
                weight_data,
                total=len(weight_data),
                ncols=100,
                desc=set_color(f"Train weight epoch {weight_epoch_idx:>5}", "pink"),
            )
            if show_progress
            else weight_data
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            optimizer.zero_grad()
            losses = self.model.weight_loss(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            optimizer.step()
        self.logger.info(
            "Train Weiht Loss " + str(weight_epoch_idx) + " loss: " + str(total_loss)
        )
        return total_loss

    def get_provider_fairness_weight(self, item_provider):
        # get the user scores of all items
        interaction = {}
        interaction["user_id"] = torch.arange(1, self.model.n_users).to(
            self.model.device
        )
        user_scores = (
            self.model.full_sort_predict(interaction)
            .reshape(-1, self.model.n_items)
            .detach()
        )
        user_scores[:, 0] = 0
        _, item_matrix = torch.topk(user_scores, 100, dim=-1)
        item_matrix = np.array(item_matrix.cpu())

        k = item_matrix.shape[1]
        user_num = item_matrix.shape[0]
        score = 1 / np.log2(np.arange(2, 2 + k))
        item_cnt = len(item_provider)

        exposure_score = np.zeros(item_cnt)
        up_exposure_score = np.zeros((user_num, np.max(item_provider) + 1))

        for idx, rec_u in enumerate(item_matrix):
            for i in range(k):
                exposure_score[rec_u[i]] += score[i]
                up_exposure_score[idx, item_provider[rec_u[i]]] += score[i]

        ui_exposure_score = up_exposure_score[
            np.expand_dims(np.arange(user_num), 1).repeat(item_cnt, axis=1),
            np.expand_dims(item_provider, 0).repeat(user_num, axis=0),
        ]

        # count item num of each provider
        provider_cnt = np.max(item_provider) + 1
        provider_num_items = np.zeros((provider_cnt,))
        for _, provider in enumerate(item_provider):
            provider_num_items[provider] += 1
        provider_num_items[0] = 1

        # calculate the exposure score of each provider
        provider_exposure_score = np.zeros((item_cnt, len(provider_num_items)))
        provider_exposure_score[np.arange(item_cnt), item_provider] = exposure_score
        provider_exposure_score = provider_exposure_score.sum(0)
        provider_exposure_score = provider_exposure_score / provider_num_items

        provider_exposure_score = pow(
            provider_exposure_score, self.config["provider_eta"]
        )
        provider_exposure_score = 1 / (provider_exposure_score + self.config["delta"])
        # provider_exposure_score = provider_exposure_score.max() - provider_exposure_score
        provider_fairness_weight = provider_exposure_score[item_provider]
        provider_fairness_weight = torch.tensor(provider_fairness_weight).to(
            self.model.device
        )
        return provider_fairness_weight, ui_exposure_score

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", "none") != "none":
            train_data.get_model(self.model)
        valid_step = 0

        num_users = train_data.dataset.num(self.config["USER_ID_FIELD"])
        user_counter = train_data.dataset.user_counter

        # get the user popularity for fairness sampling
        user_popularity = np.zeros(num_users)
        for user_id, count_user in user_counter.items():
            user_popularity[user_id] = count_user

        num_items = train_data.dataset.num(self.config["ITEM_ID_FIELD"])
        item_counter = train_data.dataset.item_counter

        # get the item popularity for fairness sampling
        item_popularity = np.zeros(num_items)
        for item_id, count_item in item_counter.items():
            item_popularity[item_id] = count_item
        item_popularity = item_popularity / item_popularity.sum()

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()

            if self.config["fairness_type"] is not None:
                item_provider = train_data.dataset.get_item_feature()[
                    self.config["PRODIVER_ID_FIELD"]
                ].numpy()
                (
                    provider_fairness_weight,
                    ui_exposure_score,
                ) = self.get_provider_fairness_weight(item_provider)
                user_fairness_weight = (
                    np.concatenate(
                        [
                            np.ones((1, ui_exposure_score.shape[1])),
                            1
                            / (
                                pow(ui_exposure_score, self.config["user_eta"])
                                + self.config["delta"]
                            ),
                        ],
                        axis=0,
                    )
                ) * self.model.rating_matrix.cpu().numpy()
                user_fairness_weight = torch.tensor(
                    user_fairness_weight
                    / (
                        user_fairness_weight.sum(axis=1, keepdims=True)
                        + self.config["delta"]
                    ).repeat(
                        train_data.dataset.num(self.config["ITEM_ID_FIELD"]), axis=1
                    )
                    * np.expand_dims(user_popularity, axis=1).repeat(
                        train_data.dataset.num(self.config["ITEM_ID_FIELD"]), axis=1
                    )
                ).to(self.device)

                provider_fairness_weight[0] = 0
                provider_fairness_weight_sum = provider_fairness_weight[
                    train_data.dataset.inter_feat[self.config["ITEM_ID_FIELD"]]
                    .cpu()
                    .numpy()
                ].sum()
                provider_fairness_weight = (
                    provider_fairness_weight
                    / provider_fairness_weight_sum
                    * len(train_data.dataset)
                )
                self.model.fairness_weight = (
                    provider_fairness_weight,
                    user_fairness_weight,
                )

                weight_data = UserDataLoader(
                    self.config, train_data.dataset, train_data.sampler, shuffle=True
                )
                for weight_epoch_idx in range(self.config["weight_epochs"]):
                    weight_loss = self.train_weight_epoch(
                        weight_epoch_idx, weight_data, show_progress=show_progress
                    )

                del provider_fairness_weight
                del user_fairness_weight

                self.model.fairness_weight = torch.zeros(num_users, num_items)
                for _, interaction in enumerate(weight_data):
                    user = interaction[self.model.USER_ID]
                    self.model.fairness_weight[user.cpu()] = (
                        self.model.encoder(self.model.rating_matrix[user].float())
                        .detach()
                        .cpu()
                    )
                train_loss = self._train_epoch(
                    train_data, epoch_idx, show_progress=show_progress
                )
            else:
                train_loss = self._train_epoch(
                    train_data, epoch_idx, show_progress=show_progress
                )

            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)

            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )
                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result
