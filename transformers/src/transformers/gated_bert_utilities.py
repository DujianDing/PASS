import torch
from torch import nn
import numpy as np

# L0 regularization
class ConcreteGate(nn.Module):
    """
    A gate made of stretched concrete distribution (using experimental Stretchable Concrete™)
    Can be applied to sparsify neural network activations or weights.
    Example usage: https://gist.github.com/justheuristic/1118a14a798b2b6d47789f7e6f511abd
    :param shape: shape of gate variable. can be broadcasted.
        e.g. if you want to apply gate to tensor [batch, length, units] over units axis,
        your shape should be [1, 1, units]
    :param temperature: concrete sigmoid temperature, should be in (0, 1] range
        lower values yield better approximation to actual discrete gate but train longer
    :param stretch_limits: min and max value of gate before it is clipped to [0, 1]
        min value should be negative in order to compute l0 penalty as in https://arxiv.org/pdf/1712.01312.pdf
        however, you can also use tf.nn.sigmoid(log_a) as regularizer if min, max = 0, 1
    :param l0_penalty: coefficient on the regularizer that minimizes l0 norm of gated value
    :param eps: a small additive value used to avoid NaNs
    """

    def __init__(self, shape, temperature=0.33, stretch_limits=(-0.1, 1.1),
                 l0_penalty=1.0, eps=1e-6):
        super(ConcreteGate, self).__init__()
        self.temperature, self.stretch_limits, self.eps = temperature, stretch_limits, eps
        self.l0_penalty = l0_penalty
        self.log_a = nn.Parameter(torch.empty(shape))
        nn.init.xavier_uniform_(self.log_a)

    def forward(self, values, is_train=None):
        """ applies gate to values, if is_train, adds regularizer to reg_collection """
        is_train = self.training if is_train is None else is_train
        gates = self.get_gates(is_train)
        return values * gates

    def get_gates(self, is_train):
        """ samples gate activations in [0, 1] interval """
        low, high = self.stretch_limits
        if is_train:
            shape = self.log_a.size()
            noise = (1 - 2*self.eps) * torch.rand(shape).to(self.log_a.device) + self.eps
            concrete = torch.sigmoid((torch.log(noise) - torch.log(1 - noise) + self.log_a) / self.temperature)
        else:
            concrete = torch.sigmoid(self.log_a)

        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = torch.clamp(stretched_concrete, 0, 1)
        return clipped_concrete

    def get_loga(self):
        return self.log_a

    def get_penalty(self):
        """
        Computes l0 and l2 penalties. For l2 penalty one must also provide the sparsified values
        (usually activations or weights) before they are multiplied by the gate
        Returns the regularizer value that should to be MINIMIZED (negative logprior)
        """
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"
        # compute p(gate_is_closed) = cdf(stretched_sigmoid < 0)
        p_open = torch.sigmoid(self.log_a - self.temperature * np.log(-low / high))
        p_open = torch.clamp(p_open, self.eps, 1.0 - self.eps)

        total_reg = self.l0_penalty * torch.sum(p_open)
        return total_reg

    def get_penalty_p1(self):
        """
        Computes penalty p1
        """
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"

        p_1 = torch.sigmoid(self.log_a - self.temperature * np.log((1 - low) / (high - 1)))
        p_1 = torch.clamp(p_1, self.eps, 1.0 - self.eps)

        total_reg = torch.sum(p_1)

        return total_reg

    def get_penalty_p0(self):
        """
        Computes penalty p0
        """
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"

        p_0 = torch.sigmoid(self.temperature * np.log(-low / high) - self.log_a)
        p_0 = torch.clamp(p_0, self.eps, 1.0 - self.eps)

        total_reg = torch.sum(p_0)

        return total_reg

    def get_penalty_concentrator(self):
        """
        Computes penalty concentrator
        """
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"

        p_0 = torch.sigmoid(self.temperature * np.log(-low / high) - self.log_a)
        p_0 = torch.clamp(p_0, self.eps, 1.0 - self.eps)

        total_reg = 1 - torch.prod(p_0)

        return total_reg

    def get_penalty_pnb(self):
        """
        Computes penalty pnb
        """
        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"

        p_0 = torch.sigmoid(self.temperature * np.log(-low / high) - self.log_a)
        p_1 = torch.sigmoid(self.log_a - self.temperature * np.log((1 - low) / (high - 1)))
        p_not_binary = 1 - (p_1 + p_0)
        p_not_binary = torch.clamp(p_not_binary, self.eps, 1.0 - self.eps)

        total_reg = torch.sum(p_not_binary)

        return total_reg

    def get_sparsity_rate(self):
        """ Computes the fraction of gates which are now active (non-zero) """
        is_nonzero = self.get_gates(False) == 0.0
        return torch.mean(is_nonzero.float())

EPSILON = torch.finfo(torch.double).tiny
