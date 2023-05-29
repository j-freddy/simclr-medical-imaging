from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResNetTransferLM(LightningModule):
    def __init__(self, backbone, num_classes, lr, momentum):
        super().__init__()

        # Save constructor parameters to self.hparams
        self.save_hyperparameters(ignore=["backbone"])

        # ResNet model
        self.backbone = backbone
        # Replace projection head with linear layer
        self.backbone.fc = nn.Linear(
            self.backbone.fc[0].in_features,
            num_classes,
        )
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.backbone.parameters(), lr=0.001)

        return optimizer
    
    def forward(self, x):
        return self.backbone(x)

    def loss(self, batch, mode="train"):
        feats, labels = batch
        preds = self.backbone(feats)

        labels = labels.squeeze()

        loss = F.cross_entropy(preds, labels.long())
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
