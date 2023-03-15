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
            # TODO Sometimes ReLU is not used
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

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        # Concatenates tensors into 1D
        imgs = torch.cat(imgs, dim=0)
        # Apply base encoder and projection head to imgs to get embedded encoders
        zs = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(zs[:, None, :], zs[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(
            cos_sim.shape[0],
            dtype=torch.bool,
            device=cos_sim.device
        )
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        # InfoNCE loss
        cos_sim /= self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)

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

        return nll

    def training_step(self, batch, batch_index):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_index):
        self.info_nce_loss(batch, mode="val")
