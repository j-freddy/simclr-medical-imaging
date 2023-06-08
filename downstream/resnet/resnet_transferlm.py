from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResNetTransferLM(LightningModule):
    def __init__(self, backbone, num_classes, lr, momentum):
        """
        ResNet-18 PyTorch Lightning module

        The base encoder of a pretrained model (ResNet-18) is extracted, and an
        extra linear layer (with cross-entropy loss) is appended to the end of
        the encoder so the output becomes a predicted label. The entire encoder
        gets finetuned during transfer learning.

        You can also use this environment to train a ResNet-18 network from
        scratch.

        Args:
            backbone (ResNet): The ResNet-18 encoder.
            num_classes (int): The number of classes in the classification task.
                This is used as the output dimension for the final linear layer.
            lr (float): The learning rate.
            momentum (float): Momentum for the optimiser. Currently deprecated
                (unused with Adam).
        """
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
        """
        Lightning Module utility method. Using Adam optimiser. Do not call this
        method.
        """
        optimizer = optim.Adam(self.backbone.parameters(), lr=0.001)
        return optimizer
    
    def forward(self, x):
        """
        Performs forward pass on the input data.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output data.
        """
        return self.backbone(x)

    def loss(self, y, y_pred):
        """
        Computes the cross-entropy loss.

        Args:
            y (torch.Tensor): The target labels.
            y_pred (torch.Tensor): The predicted labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        return F.cross_entropy(y_pred, y.long())
    
    def step(self, batch, mode="train"):
        """
        Performs a forward pass for a given batch. This method should not be
        called. Use fit() instead.
        """
        x, y = batch
        y = y.squeeze()

        y_pred = self.forward(x)

        loss = self.loss(y, y_pred)
        acc = (y_pred.argmax(dim=-1) == y).float().mean()

        self.log(mode + "_loss", loss)
        self.log(mode + "_acc", acc)

        return loss


    def training_step(self, batch, batch_index):
        """
        Performs a forward training pass for a given batch. Lightning Module
        utility method. This method should not be called. Use fit() instead.
        """
        return self.step(batch, mode="train")

    def validation_step(self, batch, batch_index):
        """
        Performs a forward validation pass for a given batch. Lightning Module
        utility method. This method should not be called. Use fit() instead.
        """
        self.step(batch, mode="val")

    def test_step(self, batch, batch_index):
        """
        Performs a forward test pass for a given batch. Lightning Module
        utility method. This method should not be called. Use fit() instead.
        """
        self.step(batch, mode="test")
