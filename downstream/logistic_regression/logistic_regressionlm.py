import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LogisticRegression(pl.LightningModule):
    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

    def configure_optimizers(self):
        # optimizer = optim.AdamW(
        #     self.parameters(),
        #     lr=self.hparams.lr,
        #     weight_decay=self.hparams.weight_decay,
        # )

        # lr_scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer,
        #     milestones=[
        #         int(self.hparams.max_epochs * 0.6),
        #         int(self.hparams.max_epochs * 0.8),
        #     ],
        #     gamma=0.1,
        # )

        # optimizer = optim.SGD(
        #     self.backbone.parameters(),
        #     lr=self.hparams.lr,
        #     momentum=self.hparams.momentum,
        # )

        # lr_scheduler = optim.lr_scheduler.StepLR(
        #     optimizer,
        #     step_size=7,
        #     gamma=0.1,
        # )

        optimizer = optim.Adam(self.backbone.parameters(), lr=0.001)

        # return [optimizer], [lr_scheduler]
        return optimizer

    def loss(self, batch, mode="train"):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(mode + "_loss", loss)
        self.log(mode + "_acc", acc)
        return loss

    def training_step(self, batch, batch_index):
        return self.loss(batch, mode="train")

    def validation_step(self, batch, batch_index):
        self.loss(batch, mode="val")

    def test_step(self, batch, batch_index):
        self.loss(batch, mode="test")
