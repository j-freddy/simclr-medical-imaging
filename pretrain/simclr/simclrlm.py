import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class SimCLRLM(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=100):
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
        return self.convnet(x)

    def loss(self, cos_sim, pos_mask):
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        return nll.mean()

    def step(self, batch, mode="train"):
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
        return self.step(batch, mode="train")

    def validation_step(self, batch, batch_index):
        self.step(batch, mode="val")
