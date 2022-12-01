import torch
import torch.nn as nn

import Models
import Data


def train_epoch(device: torch.device, designer: Models.Designer, lookout: Models.Lookout, designer_optimizer: torch.optim.AdamW, lookout_optimizer: torch.optim.AdamW, designer_loss_weights: list, mislabel_rate: float, data: torch.Tensor):

    n = data.size(0)
    r = torch.randperm(n)

    batch_size = 128
    batch_num = n // batch_size

    designer_losses = []
    lookout_losses = []
    for i in range(batch_num):

        batch_sections, batch_strips = Data.get_batch(device, data, i, batch_size, r)

        designer_loss, lookout_loss = train_batch(device, designer, lookout, designer_optimizer, lookout_optimizer, designer_loss_weights, mislabel_rate, batch_sections, batch_strips)

        designer_losses.append(designer_loss)
        lookout_losses.append(lookout_loss)


    return designer_losses, lookout_losses



def train_batch(device: torch.device, designer: Models.Designer, lookout: Models.Lookout, designer_optimizer: torch.optim.AdamW, lookout_optimizer: torch.optim.AdamW, designer_loss_weights: list, mislabel_rate: float, batch_sections: torch.Tensor, batch_strips: torch.Tensor):

    designer_loss, batch_camo_strips, batch_locs = train_designer_batch(device, designer, lookout, designer_optimizer, designer_loss_weights, batch_sections, batch_strips)
    lookout_loss = train_lookout_batch(device, lookout, lookout_optimizer, mislabel_rate, batch_camo_strips, batch_locs)

    return designer_loss, lookout_loss


def train_designer_batch(device: torch.device, designer: Models.Designer, lookout: Models.Lookout, designer_optimizer: torch.optim.AdamW, designer_loss_weights: list, batch_sections: torch.Tensor, batch_strips: torch.Tensor) -> list:

    designer_optimizer.zero_grad()

    batch_camos, mu, log_var = designer(batch_strips)
    batch_camo_strips, batch_locs = Data.apply_camo(device, batch_camos, batch_strips)
    loss = design_loss(designer_loss_weights, batch_camos, batch_sections, mu, log_var, lookout, batch_camo_strips, batch_locs)

    loss.backward()
    designer_optimizer.step()

    return loss.item(), batch_camo_strips.clone().detach(), batch_locs.clone().detach()



def train_lookout_batch(device: torch.device, lookout: Models.Lookout, lookout_optimizer: torch.optim.AdamW, mislabel_rate: float, camo_strips: torch.Tensor, locs: torch.Tensor) -> float:

    shuffle_num = int(mislabel_rate * locs.size(0))
    locs[:shuffle_num] = locs[torch.randperm(shuffle_num)]

    lookout_optimizer.zero_grad()

    preds = lookout(camo_strips)
    loss = nn.BCELoss()(preds, locs)

    loss.backward()
    lookout_optimizer.step()

    return loss.item()



def design_loss(weights, preds, targets, mu, log_var, lookout, camo_strips, locs) -> torch.Tensor:

    KLD = -0.5 * torch.sum(1 + log_var - log_var.exp() - mu.pow(2))
    RECON = nn.BCELoss()(preds, targets)
    LOOK = nn.BCELoss()(lookout(camo_strips), torch.ones_like(locs)/locs.size(1))

    loss = weights[0]*RECON + weights[1]*KLD + weights[2]*LOOK

    return loss