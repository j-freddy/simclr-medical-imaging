import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class SimCLRLM(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=100):
        """
        SimCLR PyTorch Lightning module.

        A Simple Framework for Contrastive Learning of Visual Representations
        (SimCLR) is a state-of-the-art contrastive learning method that aims to
        learn useful representations of images through training a convolutional
        neural network (the codebase uses ResNet-18) to recognise similarities
        between a pair of augmented data points derived from the same input
        image. The idea is that the network may learn to extract useful,
        generalisable features that can be used for downstream tasks.

        Original papers: See README.md in root repository

        Args:
            hidden_dim (int): Number of dimensions in the projected layer. This
                is the embedding space where we compare similarity of projected
                views.
            lr (float): The learning rate.
            temperature (float): The temperature in InfoNCE loss.
            weight_decay (float): Weight decay for AdamW optimizer.
            max_epochs (int, optional): Maximum number of training epochs.
                Defaults to 100.
        """
        super().__init__()

        # Save constructor parameters to self.hparams
        self.save_hyperparameters()

        # Base encoder
        # num_classes is output size of the last linear layer
        self.convnet = torchvision.models.resnet18(
            weights=None,
            num_classes=4 * self.hparams.hidden_dim
        )

        self.convnet.fc = nn.Sequential(
            self.convnet.fc,
            # Attach projection head
            nn.ReLU(inplace=True),
            nn.Linear(4 * self.hparams.hidden_dim, self.hparams.hidden_dim)
        )

    def configure_optimizers(self):
        """
        Lightning Module utility method. Using AdamW optimiser with
        CosineAnnealingLR scheduler. Do not call this method. 
        """
        # AdamW decouples weight decay from gradient updates
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Set learning rate using a cosine annealing schedule
        # See https://pytorch.org/docs/stable/optim.html
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.lr / 50
        )

        return [optimizer], [lr_scheduler]

    def forward(self, x):
        """
        Performs forward pass on the input data.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output data.
        """
        return self.convnet(x)

    def loss(self, cos_sim, pos_mask):
        """
        Computes the InfoNCE loss given the cosine similarity and the positive
        pair of examples.

        Args:
            cos_sim (torch.Tensor): The cosine similarity matrix.
            pos_mask (torch.Tensor): The mask to get the positive pair.

        Returns:
            torch.Tensor: The computed loss.
        """
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        return nll.mean()

    def step(self, batch, mode="train"):
        """
        Performs a forward pass for a given batch. This method should not be
        called. Use fit() instead.
        """
        x, _ = batch
        # Concatenates tensors into 1D
        x = torch.cat(x, dim=0)
        # Apply base encoder and projection head to images to get embedded
        # encoders
        z = self.forward(x)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(
            cos_sim.shape[0],
            dtype=torch.bool,
            device=cos_sim.device
        )
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        cos_sim /= self.hparams.temperature

        # InfoNCE loss
        loss = self.loss(cos_sim, pos_mask)

        # Logging loss
        self.log(mode + "_loss", loss)

        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],
            # First position positive example
            dim=-1,
        )

        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

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
