import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-Head Self Attention
class AttentionMask(nn.Module):
    #
    # Tensorflow -> Pytorch
    # __init__ removed regularizer, constraints
    # forward = build, call
    # compute_output_shape, get_config removed
    #

    def __init__(
        self,
        hidden,
        da,
        r,
        activation="tanh",
        returnAttention=False,
        attentionRegularizerWeight=0.0,
        normalize=False,
        attmod="smooth",
        sharpBeta=1,
    ):
        super(AttentionMask, self).__init__()
        self.hidden = hidden
        self.da = da
        self.r = r
        self.activation = activation
        self.returnAttention = returnAttention
        self.attentionRegularizerWeight = attentionRegularizerWeight
        self.normalize = normalize
        self.attmod = attmod
        self.sharpBeta = sharpBeta

        self.W1 = nn.Parameter(torch.Tensor(hidden, da))
        self.W2 = nn.Parameter(torch.Tensor(da, r))
        # why use xavier_uniform ?
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

        if activation == "tanh":
            self.activation = torch.tanh
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, H):
        H1 = H[:, :, :-1]  # (batch_size, n, hidden) <-> input
        attention_mask = H[:, :, -1]  # (batch_size, n, 1) <-> attention_mask

        H_t = self.activation(torch.matmul(H1, self.W1))  # (batch_size, n, da)
        temp = torch.matmul(H_t, self.W2).premute(0, 2, 1)  # (batch_size, r, n)

        # mask add on temp for softmax, for padding regions
        # mask: (batch_size, n) -> (batch_size, r, n)
        mask = (1.0 - attention_mask.float()) * -10000.0
        temp += mask.unsqueeze(1).repeat(1, self.r, 1)

        if self.attmod == "softmax":  # paper using smoothing method instead
            A = F.softmax(temp * self.sharpBeta, dim=-1)
        elif self.attmod == "smooth":  # equation (2)
            _E = torch.sigmoid(
                temp * self.sharpBeta
            )  # sigmoid all elements in Energy matrix
            sumE = _E.sum(dim=2, keepdim=True)
            A = _E / sumE

        if self.normalize:  # equation (3)
            length = attention_mask.float().sum(
                dim=1, keepdim=True
            ) / attention_mask.size(1)
            lengthr = length.unsequeeze(1).repeat(1, self.r, 1)
            A = A * lengthr

        # equation (4)
        M = torch.bmm(A, H1)  # (batch_size, r, hidden) <-> Final context emebedding

        # pytorch doesn't have add_loss, need to add munually in training loop
        if self.attentionRegularizerWeight > 0.0:
            regularization_loss = self._attention_regularizer(A)
        else:
            regularization_loss = 0.0

        if self.returnAttention:
            return M, regularization_loss, A

        return M, regularization_loss

    def _attention_regularizer(self, attention):
        #
        # AAT - I
        # Equation (5)
        #
        batch_size = attention.size(0)
        identity = torch.eye(self.r, device=attention.device)  # (r, r)
        temp = (
            torch.bmm(attention, attention.permute(0, 2, 1)) - identity
        )  # (batch_size, r, r)
        penalty = self.attentionRegularizerWeight * temp.pow(2).sum() / batch_size
        return penalty
