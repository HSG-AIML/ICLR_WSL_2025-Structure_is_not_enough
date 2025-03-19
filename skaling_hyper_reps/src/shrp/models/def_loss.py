# -*- coding: utf-8 -*-
################################################################################
# code originally take from https://github.com/Spijkervet/SimCLR/blob/master/modules/nt_xent.py
##########################
import numpy as np

import torch
import torch.nn as nn

from einops import repeat

import warnings

from torchmetrics.functional import r2_score
from torchmetrics.functional import explained_variance

from shrp.datasets.dataset_auxiliaries import tokens_to_recipe, tokens_to_checkpoint
from shrp.models.def_net import CNN, CNN3, LeNet5


class MaskedReconLoss(nn.Module):
    """
    Recon loss with masks
    """

    def __init__(self, reduction):
        super(MaskedReconLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
        self.loss_mean = None
        self.scale = None

    def forward(self, output, target, mask, _):
        """
        Args:
            output: torch.tensor of shape [batchsize,window,tokendim] with model predictions
            target: torch.tensor of shape [batchsize,window,tokendim] with model predictions
            mask: torch.tensor of shape [batchsize,window,tokendim] with maks for original non-zero values
        Returns:
            dict: "loss_recon": masekd MSE losss, "rsq": R^2 to mean
        """
        assert (
            output.shape == target.shape == mask.shape
        ), f"MSE loss error: prediction and target don't have the same shape. output {output.shape} vs target {target.shape} vs mask {mask.shape}"
        # apply mask
        output = mask * output
        loss = self.criterion(output, target)

        if self.scale is None:
            self.scale = loss.item()

        loss = loss / self.scale

        rsq = 0
        if self.loss_mean:
            rsq = torch.tensor(1 - loss.item() / self.loss_mean)
        else:
            rsq = explained_variance(
                preds=output, target=target, multioutput="uniform_average"
            )

            # rsq = r2_score(
            #     output.view(output.shape[0], -1),
            #     target.view(target.shape[0], -1),
            #     multioutput="uniform_average",
            # )
            # rsq = r2_score(output, target, multioutput="raw_values")

        return {"loss_recon": loss, "loss_structure": loss, 'loss_behaviour': torch.tensor(0.0), "rsq": rsq}

    def set_mean_loss(self, data: torch.Tensor, mask: torch.Tensor):
        """
        #TODO
        """
        # check that data are tensor..
        assert isinstance(data, torch.Tensor)
        w_mean = data.mean(dim=0)  # compute over samples (dim0)
        # scale up to same size as data
        data_mean = repeat(w_mean, "l d -> n l d", n=data.shape[0])
        out_mean = self.forward(data_mean, data, mask)

        # compute mean
        print(f" mean loss: {out_mean['loss_recon']}")

        self.loss_mean = out_mean["loss_recon"]

class DistillationLoss(nn.Module):
    def __init__(self, reference_checkpoint, reference_params, queryset="random", dump=None, n_queries=64, loss='l2', reduction="mean", temperature=2.0):
        super(DistillationLoss, self).__init__()
        self.reference_checkpoint = reference_checkpoint
        self.queryset = queryset
        self.n_queries = n_queries

        self.scale = None

        if loss == 'l2':
            self.criterion = lambda output, target: DistillationLoss.__l2_loss(output, target, reduction=reduction)
        elif loss == 'cross_entropy':
            self.criterion = lambda output, target: DistillationLoss.__ce_loss(output, target, reduction=reduction)
        elif loss == 'distillation':
            self.criterion = lambda output, target: DistillationLoss.__distillation_loss(output, target, T=temperature)
        else:
            raise ValueError(f"Unknown loss: {loss} (accepted are 'l2' or 'distillation')")

        self.recipe = None

        if reference_params['model::type'] == 'CNN':
            self.i_dim = (reference_params['model::channels_in'], 28, 28)  # 'i_dim': '(C, 28, 28)
            self.model = CNN(
                channels_in=reference_params['model::channels_in'],
                nlin=reference_params['model::nlin'],
                dropout=reference_params['model::dropout']
            )
            
        elif reference_params['model::type'] == 'CNN3':
            self.i_dim = (reference_params['model::channels_in'], 32, 32)  # 'i_dim': '(C, 32, 32)
            self.model = CNN3(
                channels_in=reference_params['model::channels_in'],
                nlin=reference_params['model::nlin'],
                dropout=reference_params['model::dropout']
            )

        elif reference_params['model::type'] == 'LeNet5':
            self.i_dim = (reference_params['model::channels_in'], 32, 32)  # 'i_dim': '(C, 32, 32)
            self.model = LeNet5(
                channels_in=reference_params['model::channels_in'],
                nlin=reference_params['model::nlin'],
                dropout=reference_params['model::dropout']
            )
            
        else:
            raise NotImplementedError(f'Unknown model: {reference_params["model::type"]}')
        
        if reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'sum':
            self.reduction = torch.sum
        else:
            raise ValueError(f"Unknown reduction: {reduction} (accepted are 'mean' or 'sum')")

        if queryset == 'union':
            if self.n_queries % 2 != 0:
                warnings.warn(f"Union queryset requires an even number of queries, {self.n_queries} is not even. Adding 1.")
                self.n_queries += 1

            self.n_queries = self.n_queries // 2
        
        if queryset in ['data', 'union']:
            if dump is None:
                raise ValueError(f"Queryset {queryset} requires a dump data file")
            else:
                self.dataloader = torch.utils.data.DataLoader(torch.load(dump)['trainset'], batch_size=self.n_queries, shuffle=True)


    def __generate_queries_random(self):
        return torch.rand(self.n_queries, *self.i_dim)
    
    def __generate_queries_data(self):
        return next(iter(self.dataloader))[0] # [0] because we don't want the labels
    
    def generate_queries(self):
        if self.queryset == 'random':
            return self.__generate_queries_random()
        elif self.queryset == 'data':
            return self.__generate_queries_data()
        elif self.queryset == 'union':
            return torch.vstack((
                self.__generate_queries_random(),
                self.__generate_queries_data()
            ))
        else:
            raise ValueError(f"Unknown queryset: {self.queryset} (accepted are 'random', 'data' or 'union)")

    def forward(self, output, target, _, pos):
        queries = self.generate_queries().to(output.device).type(output.dtype)

        if self.recipe is None:
            self.recipe = tokens_to_recipe(pos[0], self.reference_checkpoint)

        loss = torch.vmap(
            DistillationLoss.forward_elementwise,
            in_dims=(0, 0, None, None, None, None, None), 
            out_dims=0
        ) (output, target, self.model, self.reference_checkpoint, self.recipe, self.criterion, queries)

        # We filter out NaNs and Infs to avoid total loss collapse
        loss = self.reduction(loss[loss.isfinite()])

        if self.scale is None:
            self.scale = loss.item()

        loss = loss / self.scale

        return {'loss_recon': loss, 'loss_structure': torch.tensor(0.0), 'loss_behaviour': loss}
    
    @staticmethod
    def __l2_loss(output, target, reduction):
        soft_target = nn.functional.softmax(target, dim=-1)
        soft_output = nn.functional.softmax(output, dim=-1)

        soft_target = torch.nan_to_num(soft_target, nan=0.0, posinf=1.0, neginf=0.0)
        soft_output = torch.nan_to_num(soft_output, nan=0.0, posinf=1.0, neginf=0.0)

        return torch.nn.functional.mse_loss(soft_output, soft_target, reduction=reduction)

    @staticmethod
    def __ce_loss(output, target, reduction):
        soft_target = nn.functional.softmax(target, dim=-1)
        soft_target = torch.nan_to_num(soft_target, nan=0.0, posinf=1.0, neginf=0.0)

        output = torch.nan_to_num(output, nan=0.0)

        return nn.functional.cross_entropy(output, target, reduction=reduction)
    
    @staticmethod
    def __distillation_loss(output, target, T):
        soft_target = nn.functional.softmax(target / T, dim=-1)
        soft_prob = nn.functional.log_softmax(output / T, dim=-1)

        soft_target = torch.nan_to_num(soft_target, nan=1e-9, posinf=1.0, neginf=1e-9)
        soft_prob = torch.nan_to_num(soft_prob, nan=np.log(1e-9), posinf=np.log(1.0), neginf=np.log(1e-9))

        return torch.sum(soft_target * (soft_target.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

    @staticmethod
    def forward_elementwise(output, target, model, reference_checkpoint, recipe, criterion, queries):
        with torch.no_grad():
            target_checkpoint = tokens_to_checkpoint(target, None, reference_checkpoint, recipe=recipe)
            target = model.forward_with_state_dict(queries, target_checkpoint)

        output_checkpoint = tokens_to_checkpoint(output, None, reference_checkpoint, recipe=recipe)
        output = model.forward_with_state_dict(queries, output_checkpoint)

        return criterion(output, target)

class ReconDistillationLoss(nn.Module):
    def __init__(self, reference_checkpoint, reference_params, beta=0.1, reduction='mean', queryset="random", dump=None, n_queries=64, loss='l2', temperature=2.0):
        super().__init__()
        self.recon_loss = MaskedReconLoss(reduction=reduction)
        self.distillation_loss = DistillationLoss(reference_checkpoint, reference_params, queryset=queryset, dump=dump, n_queries=n_queries, loss=loss, temperature=temperature)
        self.beta = beta

    def forward(self, output, target, mask, pos):
        recon_loss = self.recon_loss(output, target, mask, pos)
        distillation_loss = self.distillation_loss(output, target, mask, pos) 

        loss = self.beta * recon_loss['loss_recon'] + (1 - self.beta) * distillation_loss['loss_recon']

        return {'loss_recon': loss, 'loss_structure': recon_loss['loss_recon'], 'loss_behaviour': distillation_loss['loss_recon'], 'rsq': recon_loss['rsq']}

################################################################################################
# contrastive loss
################################################################################################
class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature): 
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.scale = None

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        # create mask for negative samples: main diagonal, +-batch_size off-diagonal are set to 0
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: representations of batch in two different views. shape: batch_size x C
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # dimension of similarity matrix
        N = 2 * self.batch_size
        # concat both representations to easily compute similarity matrix
        z = torch.cat((z_i, z_j), dim=0)
        # compute similarity matrix around dimension 2, which is the representation depth. the unsqueeze ensures the matmul/ outer product
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # take positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples,resulting in: 2xNx1
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative samples are singled out with the mask
        negative_samples = sim[self.mask].reshape(N, -1)

        # reformulate everything in terms of CrossEntropyLoss: https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
        # labels in nominator, logits in denominator
        # positve class: 0 - that's the first component of the logits corresponding to the positive samples
        labels = torch.zeros(N).to(positive_samples.device).long()
        # the logits are NxN (N+1?) predictions for imaginary classes.
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        if self.scale is None:
            self.scale = loss.item()

        loss = loss / self.scale

        return loss


class NT_Xent_pos(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent_pos, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.scale = None

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.MSELoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        # create mask for negative samples: main diagonal, +-batch_size off-diagonal are set to 0
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: representations of batch in two different views. shape: batch_size x C
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # dimension of similarity matrix
        N = 2 * self.batch_size
        # concat both representations to easily compute similarity matrix
        z = torch.cat((z_i, z_j), dim=0)
        # compute similarity matrix around dimension 2, which is the representation depth. the unsqueeze ensures the matmul/ outer product
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # take positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples,resulting in: 2xNx1
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative samples are singled out with the mask
        # negative_samples = sim[self.mask].reshape(N, -1)

        # reformulate everything in terms of CrossEntropyLoss: https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
        # labels in nominator, logits in denominator
        # positve class: 0 - that's the first component of the logits corresponding to the positive samples
        labels = torch.zeros(N).to(positive_samples.device).unsqueeze(dim=1)
        # just minimize the distance of positive samples to zero
        loss = self.criterion(positive_samples, labels)
        loss /= N

        if self.scale is None:
            self.scale = loss.item()

        loss = loss / self.scale

        return loss


################################################################################################
# contrastive + recon loss combination
################################################################################################
class GammaContrastReconLoss(nn.Module):
    """
    #TODO docstring
    Combines NTXent Loss with reconstruction loss.
    L = gamma*NTXentLoss + (1-gamma)*ReconstructionLoss
    """

    def __init__(
        self,
        gamma: float,
        reduction: str,
        batch_size: int,
        temperature: float,
        contrast="simclr",
        z_var_penalty: float = 0.0,
        z_norm_penalty: float = 0.0,
        beta: float = 0.1,
        loss_distillation: str = "l2",
        temperature_distillation: float = 2.0,
        queryset_distillation: str = "random",
        queryset_dump: str = None,
        n_queries_distillation: int = 64,
        reference_checkpoint=None,
        reference_params=None,
        lambda_: float = 0.0
    ) -> None:
        super(GammaContrastReconLoss, self).__init__()
        # test for allowable gamma values
        assert 0 <= gamma <= 1
        self.gamma = gamma

        self.lambda_ = lambda_

        # z_var penalty
        self.z_var_penalty = z_var_penalty
        # z_norm penalty
        self.z_norm_penalty = z_norm_penalty

        # set contrast
        if contrast == "simclr":
            print("model: use simclr NT_Xent loss")
            self.loss_contrast = NT_Xent(batch_size, temperature)
        elif contrast == "positive":
            print("model: use only positive contrast loss")
            self.loss_contrast = NT_Xent_pos(batch_size, temperature)
        else:
            print("unrecognized contrast - use reconstruction only")

        assert 0 <= beta <= 1
        if beta >= 1 - 1e-10:
            self.loss_recon = MaskedReconLoss(
                reduction=reduction,
            )
        elif beta <= 1e-10:
            self.loss_recon = DistillationLoss(
                reference_checkpoint=reference_checkpoint,
                reference_params=reference_params,
                queryset=queryset_distillation,
                dump=queryset_dump,
                n_queries=n_queries_distillation,
                reduction=reduction,
                temperature=temperature_distillation,
                loss=loss_distillation
            )
        else:
            self.loss_recon = ReconDistillationLoss(
                reference_checkpoint=reference_checkpoint,
                reference_params=reference_params,
                beta=beta,
                reduction=reduction,
                queryset=queryset_distillation,
                dump=queryset_dump,
                n_queries=n_queries_distillation,
                temperature=temperature_distillation,
                loss=loss_distillation
            )

        self.loss_mean = None

    def set_mean_loss(self, weights: torch.Tensor, mask=None) -> None:
        """
        Helper function to set mean loss in reconstruction loss
        """
        # if mask not set, set it to all ones
        if mask is None:
            mask = torch.ones(weights.shape)
        # call mean_loss function
        self.loss_recon.set_mean_loss(weights, mask=mask)

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        m: torch.Tensor,
        p: torch.Tensor,
        z: torch.Tensor,
        z_hat: torch.Tensor,
    ) -> dict:
        """
        Args:
            z_i, z_j are the two different views of the same batch encoded in the representation space. dim: batch_sizexrepresentation space
            y: reconstruction. dim: batch_sizexinput_size
            t: target dim: batch_sizexinput_size
            m: mask 1 where inputs are nonezero, 0 otherwise
        Returns:
            dict with "loss" as main aggregated loss key, as well as loss / rsq components
        """
        if self.gamma < 1e-10:
            out_recon = self.loss_recon(y, t, m, p)
            out = {
                "loss/loss": out_recon["loss_recon"],
                "loss/loss_contrast": torch.tensor(0.0),
                "loss/loss_recon": out_recon["loss_recon"],
                "loss/loss_structure": out_recon["loss_structure"],
                "loss/loss_behaviour": out_recon["loss_behaviour"]
            }
            for key in out_recon.keys():
                new_key = f"loss/{key}"
                if new_key not in out:
                    out[new_key] = out_recon[key]
        elif abs(1.0 - self.gamma) < 1e-10:
            loss_contrast = self.loss_contrast(z_i, z_j)
            out = {
                "loss/loss": loss_contrast,
                "loss/loss_contrast": loss_contrast,
                "loss/loss_recon": torch.tensor(0.0),
                "loss/loss_structure": torch.tensor(0.0),
                "loss/loss_behaviour": torch.tensor(0.0)
            }
        else:
            # combine loss components
            loss_contrast = self.loss_contrast(z_i, z_j)
            out_recon = self.loss_recon(y, t, m, p)
            loss = (
                self.gamma * loss_contrast + (1 - self.gamma) * out_recon["loss_recon"]
            )
            out = {
                "loss/loss": loss,
                "loss/loss_contrast": loss_contrast,
                "loss/loss_recon": out_recon["loss_recon"],
                "loss/loss_structure": out_recon["loss_structure"],
                "loss/loss_behaviour": out_recon["loss_behaviour"]
            }
            for key in out_recon.keys():
                new_key = f"loss/{key}"
                if new_key not in out:
                    out[new_key] = out_recon[key]
                    # compute embedding properties

        if self.lambda_ > 0:
            out["loss/loss"] = out["loss/loss"] + self.lambda_ * torch.mean(
                torch.norm(z - z_hat, dim=1)
            )

        z_norm = torch.linalg.norm(z_i.view(z_i.shape[0], -1), ord=2, dim=1).mean()
        z_var = torch.mean(torch.var(z_i.view(z_i.shape[0], -1), dim=0))
        out["debug/z_norm"] = z_norm
        out["debug/z_var"] = z_var
        # if self.z_var_penalty > 0:
        out["loss/loss"] = out["loss/loss"] + self.z_var_penalty * z_var
        # if self.z_norm_penalty > 0:
        out["loss/loss"] = out["loss/loss"] + self.z_norm_penalty * z_norm

        return out
