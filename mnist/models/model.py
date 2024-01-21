from __future__ import print_function

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torchmetrics import Accuracy, F1Score, Precision, Recall


class LightMNISTClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super(LightMNISTClassifier, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.loss_history_train = np.array([])
        self.prediction_train = []
        self.correct_train = []
        self.loss_history_val = np.array([])
        self.prediction_val = []
        self.correct_val = []
        self.conv1 = nn.Conv2d(self.cfg.model.in_channels, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, self.cfg.model.out_channels)
        # metrics
        self.accuracy = Accuracy("multiclass", num_classes=self.cfg.model.out_channels)
        self.precision = Precision("multiclass", num_classes=self.cfg.model.out_channels)
        self.recall = Recall("multiclass", num_classes=self.cfg.model.out_channels)
        self.f1 = F1Score("multiclass", num_classes=self.cfg.model.out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits

    # define the loss function
    def criterion(self, logits, targets):
        return F.cross_entropy(logits, targets)

    # process inside the training loop
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)

        pred = outputs.argmax(dim=1)
        acc = self.accuracy(pred, targets)
        precision = self.precision(pred, targets)
        recall = self.recall(pred, targets)
        f1 = self.f1(pred, targets)
        self.loss_history_train = np.append(
            self.loss_history_train, loss.detach().cpu().numpy()
        )
        self.prediction_train.extend(pred[:].detach().cpu().numpy())
        self.correct_train.extend(targets[:].detach().cpu().numpy())
        # inbuilt logs
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        self.log("train_precision", precision, on_epoch=True)
        self.log("train_recall", recall, on_epoch=True)
        self.log("train_f1", f1, on_epoch=True)
        return {"loss": loss, "train_acc": acc}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        preds = self(batch)
        return preds.argmax(1)

    def on_train_epoch_end(self):
        # Print metrics for epoch
        # print('Конец тренировки')
        train_loss = self.loss_history_train.mean()
        clf_report = classification_report(
            self.correct_train, self.prediction_train, zero_division=0
        )
        print(f"\nTrain loss: {train_loss}")
        train_accuracy = self.accuracy(
            torch.Tensor(self.prediction_train), torch.Tensor(self.correct_train)
        )
        print(f"Train accuracy: {train_accuracy}")
        # print(f"Train accuracy: {self.accuracy(torch.Tensor(self.prediction_train),
        #                                      torch.Tensor(self.correct_train))}")
        print("Classification report for train:")
        print(clf_report)
        # Clear the memory for the next epoch
        self.loss_history_train = np.array([])
        self.prediction_train.clear()
        self.correct_train.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        features, labels = batch
        outputs = self(features)
        loss = self.criterion(outputs, labels)
        preds = outputs.argmax(1)
        acc = self.accuracy(preds, labels)
        precision = self.precision(preds, labels)
        recall = self.recall(preds, labels)
        f1 = self.f1(preds, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True)
        self.log("val_precision", precision, on_epoch=True)
        self.log("val_recall", recall, on_epoch=True)
        self.log("val_f1", f1, on_epoch=True)
        self.loss_history_val = np.append(
            self.loss_history_val, loss.detach().cpu().numpy()
        )
        self.prediction_val.extend(preds[:].detach().cpu().numpy())
        self.correct_val.extend(labels[:].detach().cpu().numpy())
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        # Print metrics for epoch
        val_loss = self.loss_history_val.mean()
        # print('Конец валидации')
        clf_report = classification_report(
            self.correct_val, self.prediction_val, zero_division=0
        )
        print(f"\nValidation loss: {val_loss}")
        val_accuracy = self.accuracy(
            torch.Tensor(self.prediction_val), torch.Tensor(self.correct_val)
        )
        print(f"Validation accuracy: {val_accuracy}")
        print("Classification report for validation:")
        print(clf_report)
        # Clear the memory for the next epoch
        self.loss_history_val = np.array([])
        self.prediction_val.clear()
        self.correct_val.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        features, labels = batch
        pred = self(features)
        loss = self.criterion(pred, labels)
        preds = pred.argmax(1)
        acc = self.accuracy(preds, labels)
        precision = self.precision(pred, labels)
        recall = self.recall(pred, labels)
        f1 = self.f1(pred, labels)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        self.log("test_correct", acc, on_step=True, on_epoch=True)
        self.log("test_precision", precision, on_epoch=True)
        self.log("test_recall", recall, on_epoch=True)
        self.log("test_f1", f1, on_epoch=True)
        return {"test_loss": loss, "test_acc": acc, "preds": pred, "labels": labels}

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
            momentum=self.cfg.training.momentum,
            nesterov=self.cfg.training.nesterov,
            weight_decay=self.cfg.training.weight_decay,
        )
