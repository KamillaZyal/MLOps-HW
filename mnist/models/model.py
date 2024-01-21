from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torchmetrics import Accuracy
from sklearn.metrics import classification_report, accuracy_score


class LightMNISTClassifier(pl.LightningModule):
    def __init__(self, cfg):
        super(LightMNISTClassifier, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.loss_history = np.array([])
        self.prediction = []
        self.correct = []
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
        self.accuracy = Accuracy("multiclass", num_classes=self.cfg.model.out_channels)

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
        self.loss_history = np.append(self.loss_history, loss.detach().cpu().numpy())
        self.prediction.extend(pred[:].detach().cpu().numpy())
        self.correct.extend(targets[:].detach().cpu().numpy())
        # inbuilt logs
        self.log("train_loss", loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)
        return {"loss": loss, "acc": acc}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        preds = self(batch)
        return preds.argmax(1)

    def on_train_epoch_end(self):
        # Print metrics for epoch
        train_loss = self.loss_history.mean()
        clf_report = classification_report(self.correct, self.prediction, zero_division=0)
        print(f"Train loss: {train_loss}")
        print(f"Accuracy: {accuracy_score(self.prediction,self.correct)}")
        print("Classification report:")
        print(clf_report)
        # Clear the memory for the next epoch
        self.loss_history = np.array([])
        self.prediction.clear()
        self.correct.clear()

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
            momentum=self.cfg.training.momentum,
            nesterov=self.cfg.training.nesterov,
            weight_decay=self.cfg.training.weight_decay,
        )
