from lightning.pytorch import LightningModule
import torch.nn as nn
import torch
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.functional import accuracy
import torch.nn.functional as F


class LitModel(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mcf1s = nn.ModuleDict({'f1':MulticlassF1Score(num_classes=25, average='macro')})

    def training_step(self, batch, batch_idx):
        x = batch['pixel_values']
        y = batch['label']
        output = self.model(x)

        loss = F.cross_entropy(output, y)
        acc = accuracy(torch.argmax(output, dim=1), y, task="multiclass", num_classes=25)
        f1 = self.mcf1s['f1'](torch.argmax(output, dim=1), y)

        self.log('loss', loss, on_epoch=True)
        self.log('acc', acc, on_epoch=True)
        self.log('f1', f1, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-5, momentum=0.9)

        return optimizer

    def validation_step(self, batch, batch_idx):
        x = batch['pixel_values']
        y = batch['label']
        output = self.model(x)

        val_loss = F.cross_entropy(output, y)
        val_acc = accuracy(torch.argmax(output, dim=1), y, task="multiclass", num_classes=25)
        val_f1 = self.mcf1s['f1'](torch.argmax(output, dim=1), y)

        self.log('val_loss', val_loss, on_epoch=True)
        self.log('val_acc', val_acc, on_epoch=True)
        self.log('val_f1', val_f1, on_epoch=True)

        return val_loss

    def predict_step(self, batch, batch_idx=None):
        x = batch
        output = self.model(x)

        return output