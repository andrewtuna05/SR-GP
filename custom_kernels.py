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

class GeneralizedCauchyKernel(Kernel):
    """
    Generalized Cauchy kernel (stationary).

    .. math::
        k(r) = \left(1 + r^\alpha\right)^{-\beta/\alpha}, \quad r = \|x-x'\|/\ell

    Trainable Parameters:
    - alpha ∈ (0, 2] (controls shape)
    - beta > 0 (controls decay)
    - lengthscale ℓ can be scalar or vector (via `ard_num_dims`)
    """
    has_lengthscale = True
    is_stationary = True

    def __init__(
        self,
        alpha_constraint=None,
        beta_constraint=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        # α constraint: (0,2]
        if alpha_constraint is None:
            alpha_constraint = Interval(1e-5, 2.0)
        self.register_parameter(
            name="raw_alpha",
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_alpha", alpha_constraint)

        # β constraint: > 0
        if beta_constraint is None:
            beta_constraint = Positive()
        self.register_parameter(
            name="raw_beta",
            parameter=torch.nn.Parameter(torch.tensor(1.0))
        )
        self.register_constraint("raw_beta", beta_constraint)

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        self.initialize(
            raw_alpha=self.raw_alpha_constraint.inverse_transform(torch.as_tensor(value))
        )
    
    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        self.initialize(
            raw_beta=self.raw_beta_constraint.inverse_transform(torch.as_tensor(value))
        )
        
    def forward(self, x1, x2, diag=False, **params):
        # rescale by lengthscale
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)

        # pairwise distance r = ||x1 - x2|| / ℓ
        r = self.covar_dist(x1_, x2_, diag=diag, square_dist=False, **params)

        # base power: (1 + r^α)^(-β/α - 1)
        base = (1 + r.pow(self.alpha)).pow(-self.beta/self.alpha - 1)

        # modification factor: 1 + (1 - β) r^α
        mod = 1 + (1 - self.beta) * r.pow(self.alpha)

        return base * mod

