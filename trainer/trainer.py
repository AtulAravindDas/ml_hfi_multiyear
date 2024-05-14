"""
Trainer modules for pytorch models.

Classes
---------
Trainer(base.base_trainer.BaseTrainer)

"""

import numpy as np
import torch
from base.base_trainer import BaseTrainer
from utils import utils
import time


class Trainer(BaseTrainer):
    """
    Trainer class

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    criterion : torch.nn.Module
        The loss function.
    metric_funcs : list of callable
        List of metric functions to evaluate the model's performance.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    max_epochs : int
        The maximum number of training epochs.
    data_loader : torch.utils.data.DataLoader
        The data loader for the training data.
    validation_data_loader : torch.utils.data.DataLoader
        The data loader for the validation data.
    device : torch.device
        The device to run the training on.
    config : dict
        Configuration parameters for the trainer.

    Attributes
    ----------
    config : dict
        Configuration parameters for the trainer.
    device : torch.device
        The device to run the training on.
    data_loader : torch.utils.data.DataLoader
        The data loader for the training data.
    validation_data_loader : torch.utils.data.DataLoader
        The data loader for the validation data.
    do_validation : bool
        Flag indicating whether to perform validation during training.

    Methods
    -------
    _train_epoch(epoch)
        Training logic for an epoch.
    _validation_epoch(epoch)
        Validation logic after training an epoch.
    """

    def __init__(
        self,
        model,
        criterion,
        metric_funcs,
        optimizer,
        scheduler,
        max_epochs,
        data_loader,
        validation_data_loader,
        device,
        config,
    ):
        super().__init__(
            model,
            criterion,
            metric_funcs,
            optimizer,
            scheduler,
            max_epochs,
            config,
        )
        self.config = config
        self.device = device

        self.data_loader = data_loader
        self.validation_data_loader = validation_data_loader

        self.do_validation = True

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        Parameters
        ----------
        epoch : int
            Current training epoch.

        Returns
        -------
        None.
        """

        self.model.train()
        self.batch_log.reset()

        self.data_loader.dataset.fill_filecache(self.config["trainer"]["n_repeat_tile"])
        self.data_loader.dataset.rng.shuffle(self.data_loader.dataset.filecache)

        for batch_idx, (data, target) in enumerate(self.data_loader):

            if batch_idx == self.config["trainer"]["max_batches"]:
                break

            # Move data to device
            input, target = (
                data.to(self.device),
                target.to(self.device),
            )

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            output = self.model(input)

            # Compute the loss and its gradients
            loss = self.criterion(output, target)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Log the results
            self.batch_log.update("batch", batch_idx)
            self.batch_log.update("loss", loss.item())
            for met in self.metric_funcs:
                self.batch_log.update(met.__name__, met(output, target))

            if len(self.data_loader.dataset.filecache) == 0:
                break

        # Save the model at the end of the epoch
        utils.save_torch_model(self.model, self.config, epoch=epoch)

        # Run validation
        if self.do_validation:
            # tval = time.time()
            self._validation_epoch(epoch)
            # print(f"  validation took {time.time() - tval:4.4f}s")

    def _validation_epoch(self, epoch):
        """
        Validate after training an epoch

        Parameters
        ----------
        epoch : int
            Current training epoch.

        Returns
        -------
        None.
        """
        self.model.eval()
        self.validation_data_loader.dataset.fill_filecache(
            self.config["trainer"]["val_n_repeat_tile"]
        )

        for batch_idx, (data, target) in enumerate(self.validation_data_loader):

            input, target = (
                data.to(self.device),
                target.to(self.device),
            )

            output = self.model(input)
            loss = self.criterion(output, target)

            # Log the results
            self.batch_log.update("val_loss", loss.item())
            for met in self.metric_funcs:
                self.batch_log.update("val_" + met.__name__, met(output, target))

            # break if have reached the max number of batches for validation
            if (
                batch_idx >= self.config["trainer"]["val_max_batches"]
                or len(self.validation_data_loader.dataset.filecache) == 0
            ):
                break
