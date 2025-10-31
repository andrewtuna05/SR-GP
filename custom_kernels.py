import torch
from torch import nn
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive, Interval

class RBFNonSeparableKernel(Kernel):
    """
    Custom RBF Kernel using a non-separable lengthscale matrix A^{-1}.
    Designed for 2D inputs with parameters a, c, and rho.
    """
    has_lengthscale = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(name="raw_a", parameter=nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_parameter(name="raw_c", parameter=nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_parameter(name="raw_rho", parameter=nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_constraint("raw_a", Positive())
        self.register_constraint("raw_c", Positive())
        self.register_constraint("raw_rho", Interval(-1 + 1e-5, 1 - 1e-5))

    @property
    def a(self):
        return self.raw_a_constraint.transform(self.raw_a)
    
    @property
    def c(self):
        return self.raw_c_constraint.transform(self.raw_c)
    
    @property
    def rho(self):
        return self.raw_rho_constraint.transform(self.raw_rho)

    def forward(self, x1, x2, diag=False, **params):
        a = self.a.squeeze().to(dtype=x1.dtype, device=x1.device)
        c = self.c.squeeze().to(dtype=x1.dtype, device=x1.device)
        rho = self.rho.squeeze().to(dtype=x1.dtype, device=x1.device)

        sqrt_ac = torch.sqrt(a * c)
        A_inv = torch.tensor([[a, -rho * sqrt_ac],
                              [-rho * sqrt_ac, c]], dtype=x1.dtype, device=x1.device)

        if diag:
            return torch.ones(x1.size(0), device=x1.device, dtype=x1.dtype)

        diff = x1.unsqueeze(1) - x2.unsqueeze(0)
        n, m, _ = diff.shape
        diff_flat = diff.reshape(-1, 2)
        diff_Ainv_flat = diff_flat @ A_inv
        sq_dist_flat = torch.sum(diff_flat * diff_Ainv_flat, dim=1)
        sq_dist = sq_dist_flat.view(n, m)

        return torch.exp(-0.5 * sq_dist)
