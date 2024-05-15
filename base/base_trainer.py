"""
Base trainer modules for PyTorch models.

Classes
---------
BaseTrainer:
    Base class for all trainers.
EarlyStopping:
    Base class for early stopping.

Methods
-------
fit()
    Full training logic
_train_epoch()
    Train an epoch
_validation_epoch()
    Validate after training an epoch
check_early_stop(epoch, validation_loss, model)
    Checks if early stopping criteria are met.

"""

from abc import abstractmethod
import time
import copy
from utils.utils import MetricTracker

import torch


class BaseTrainer:
    """
    Base class for all trainers.

    This class provides the base functionality for training PyTorch models. It includes methods for
    initializing the trainer, fitting the model, and training and validation epochs.

    Methods
    -------
    fit()
        Full training logic

    _train_epoch()
        Train an epoch

    _validation_epoch()
        Validate after training an epoch

    """

    def __init__(
        self, model, criterion, metric_funcs, optimizer, scheduler, max_epochs, config
    ):
        """
        Initializes the BaseTrainer.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to be trained.
        criterion : torch.nn.Module
            The loss function used for training.
        metric_funcs : list of callable
            List of metric functions to evaluate during training.
        optimizer : torch.optim.Optimizer
            The optimizer used for training.
        max_epochs : int
            The maximum number of epochs to train the model.
        config : dict
            Configuration parameters for the trainer.

        """
        self.config = config

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.max_epochs = max_epochs
        self.early_stopper = EarlyStopping(
            **config["trainer"]["early_stopping"]["args"]
        )

        self.metric_funcs = metric_funcs
        self.batch_log = MetricTracker(
            "batch",
            "loss",
            "val_loss",
            *[m.__name__ for m in self.metric_funcs],
            *["val_" + m.__name__ for m in self.metric_funcs],
        )
        self.log = MetricTracker(
            "epoch",
            "loss",
            "val_loss",
            *[m.__name__ for m in self.metric_funcs],
            *["val_" + m.__name__ for m in self.metric_funcs],
        )

    def fit(self):
        """
        Full training logic

        This method performs the full training logic, including training and validation epochs, logging the results, and early stopping.

        """
        for epoch in range(self.max_epochs + 1):
            start_time = time.time()

            self._train_epoch(epoch)

            # log the results of the epoch
            self.batch_log.result()
            self.log.update("epoch", epoch)
            for key in self.batch_log.history:
                self.log.update(key, self.batch_log.history[key])

            # early stopping
            if self.early_stopper.check_early_stop(
                epoch, self.log.history["val_loss"][epoch], self.model
            ):
                print(
                    f"Restoring model weights from the end of the best epoch {self.early_stopper.best_epoch}: "
                    f"val_loss = {self.early_stopper.min_validation_loss:.5f}"
                )
                self.log.print(idx=self.early_stopper.best_epoch)

                self.model.load_state_dict(self.early_stopper.best_model_state)
                self.model.eval()

                break

            # Print out progress during training
            end_time = time.time()
            elapsed_time = end_time - start_time
            current_time = time.strftime("%H:%M:%S", time.localtime())
            progtext = (
                f"{current_time}  Epoch {epoch}/{self.max_epochs:2d}\n"
                + f"  {elapsed_time:.1f}s"
                + f" - lr: {self.scheduler.get_last_lr()[0]:.2e}"
            )
            for key in self.log.history:
                if key == "epoch":
                    continue
                progtext += f" - {key}: {self.log.history[key][epoch]:.5f}"
            print(progtext)

            # update the learning rate
            # self.scheduler.step(self.log.history["val_loss"][epoch])
            self.scheduler.step()

            # reset the batch_log
            self.batch_log.reset()

    @abstractmethod
    def _train_epoch(self):
        """
        Train an epoch

        This method is called to train a single epoch of the model.

        """
        raise NotImplementedError

    @abstractmethod
    def _validation_epoch(self):
        """
        Validate after training an epoch

        This method is called to perform validation after training a single epoch of the model.

        """
        raise NotImplementedError


class EarlyStopping:
    """
    Base class for early stopping.

    This class provides the functionality for early stopping during training.

    Methods
    -------
    check_early_stop(epoch, validation_loss, model)
        Checks if early stopping criteria are met.

    """

    def __init__(self, patience=1, min_delta=0):
        """
        Initializes the EarlyStopping.

        Parameters
        ----------
        patience : int, optional
            The number of epochs to wait for improvement before stopping the training. Default is 1.
        min_delta : float, optional
            The minimum change in the validation loss required to be considered as an improvement. Default is 0.

        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_model_state = None
        self.best_epoch = None

    def check_early_stop(self, epoch, validation_loss, model):
        """
        Checks if early stopping criteria are met.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        validation_loss : float
            The validation loss at the current epoch.
        model : torch.nn.Module
            The PyTorch model.

        Returns
        -------
        bool
            True if early stopping criteria are met, False otherwise.

        """
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0

            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
