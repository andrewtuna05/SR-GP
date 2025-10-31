from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

import gpytorch
from gpytorch.constraints import Positive, GreaterThan
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# Device & dtype utilities
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device():
    """Return the global default torch device (CPU or CUDA)."""
    return DEVICE

# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def to_numpy(tensor: torch.Tensor):
    """Detach (if needed), move to CPU, convert to NumPy array."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input to to_numpy must be a torch.Tensor.")
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy()


def to_torch(x, *, device=None, dtype=None):
    """Convert array‑like `x` to a torch.Tensor on the requested device/dtype."""
    device = get_device() if device is None else device
    dtype = torch.get_default_dtype() if dtype is None else dtype

    # Early exit for existing tensors
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)

    # NumPy / pandas
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=device, dtype=dtype)

    # Python scalars / lists / tuples
    return torch.tensor(x, device=device, dtype=dtype)

# -----------------------------------------------------------------------------
# Optional global GPyTorch settings (unchanged API)
# -----------------------------------------------------------------------------

def set_gpytorch_settings(dtype=torch.float64, use_cuda: bool = True):
    """Optional knob‑twiddling.  Code is device‑agnostic without this call."""
    gpytorch.settings.fast_computations.covar_root_decomposition._set_state(False)
    gpytorch.settings.fast_computations.log_prob._set_state(False)
    gpytorch.settings.fast_computations.solves._set_state(False)
    gpytorch.settings.cholesky_max_tries._set_value(100)
    gpytorch.settings.debug._set_state(False)
    gpytorch.settings.min_fixed_noise._set_value(1e-7, 1e-7, 1e-7)

    torch.set_default_dtype(dtype)

    if use_cuda and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    elif use_cuda and not torch.cuda.is_available():
        import warnings
        warnings.warn("CUDA requested but not available — running on CPU.")

    gpytorch.settings.ciq_samples._set_state(False)
    gpytorch.settings.skip_logdet_forward._set_state(False)
    gpytorch.settings.num_trace_samples._set_value(0)
    gpytorch.settings.num_gauss_hermite_locs._set_value(300)
    gpytorch.settings.num_likelihood_samples._set_value(300)
    gpytorch.settings.deterministic_probes._set_state(True)

# -----------------------------------------------------------------------------
# Scaling utilities
# -----------------------------------------------------------------------------
class MinMaxScaler:
    """Min‑max scaling to [0,1] along each feature column."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mins: torch.Tensor | None = None
        self.maxs: torch.Tensor | None = None

    # --------------------------- public API ---------------------------
    def fit(self, X):
        self.reset()
        X = to_torch(X)
        self.mins = torch.min(X, dim=0).values
        self.maxs = torch.max(X, dim=0).values
        return self.scale(X)

    def scale(self, X):
        X = to_torch(X)
        return (X - self.mins) / (self.maxs - self.mins)

    def unscale(self, X_scaled):
        X_scaled = to_torch(X_scaled)
        X = X_scaled.unsqueeze(-1) * (self.maxs - self.mins) + self.mins
        return X.squeeze(-1)


class NoScale:
    """No‑op scaler with the same interface as MinMaxScaler."""

    @staticmethod
    def fit(_):
        pass

    @staticmethod
    def scale(X):
        return to_torch(X)

    @staticmethod
    def unscale(X_scaled):
        return to_torch(X_scaled)

# -----------------------------------------------------------------------------
# GP Regression model
# -----------------------------------------------------------------------------
class GPRModel(gpytorch.models.ExactGP):
    """Exact GP regression with optional feature scaling and diagnostic tools."""

    # -------------------------------- initialisation -------------------------
    def __init__(
        self,
        train_x=None,
        train_y=None,
        *,
        kernel: gpytorch.kernels.Kernel,
        mean: str | gpytorch.means.Mean = "constant",
        scale_x: bool = True,
        optimizer: torch.optim.Optimizer | None = None
    ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-9))

        # Scaling utility
        self.scaler = MinMaxScaler() if scale_x else NoScale()
        self.optimizer = optimizer

        # Prepare training data (may be None => pure prior GP)
        if train_x is None and train_y is None:
            x_scaled = None
            y_torch = None
        else:
            x_torch = to_torch(train_x)
            y_torch = to_torch(train_y)
            self.scaler.fit(x_torch)
            x_scaled = self.scaler.scale(x_torch)

        self.train_x = train_x if train_x is None else x_torch
        self.train_y = train_y if train_y is None else y_torch
        self.train_x_scaled = x_scaled

        super().__init__(x_scaled, y_torch, likelihood)

        # Mean module
        if mean == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
        elif mean is None:
            self.mean_module = gpytorch.means.ZeroMean()
        elif isinstance(mean, gpytorch.means.Mean):
            self.mean_module = deepcopy(mean)
        else:
            raise TypeError("'mean' must be 'constant', None, or a gpytorch.means.Mean.")

        # Kernel
        self.kernel = deepcopy(kernel)

        # Finalise device placement
        self.to(get_device())

        # ----------------- diagnostics containers -----------------
        self.param_history: Dict[str, List[Any]] = {"loss": []}
        self._bic: float | None = None  # cache
        self._loocv: float | None = None

    # -------------------------------- private helpers ------------------------
    def _store_constrained_params(self, loss: torch.Tensor):
        """Log *constrained* hyper‑parameter values for diagnostics."""
        self.param_history["loss"].append(loss.item())

        named_modules = dict(self.named_modules())
        for full_name, param in self.named_parameters():
            *module_path_parts, param_name = full_name.split(".")
            module_path = ".".join(module_path_parts)
            module = named_modules[module_path] if module_path else self

            if param_name.startswith("raw_"):
                clean_name = param_name[4:]
                constrained_val = getattr(module, clean_name)
            else:
                clean_name = param_name
                constrained_val = param

            # Convert to CPU python scalar / numpy for easy serialisation.
            if constrained_val.numel() == 1:
                value: Any = constrained_val.item()
            else:
                value = constrained_val.detach().cpu().clone().numpy()

            key = f"{module_path}.{clean_name}".lstrip(".")
            self.param_history.setdefault(key, []).append(value)

    # -------------------------------- forward -------------------------------
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # -------------------------------- fitting -------------------------------
    def fit_model(
        self,
        training_iterations: int = 50,
        lr: float = 0.1,
        verbose: bool = True,
        *,
        reset_opt: bool = True,          # <-- new flag
    ):
        """Maximum-likelihood training with Adam (warm-start capable)."""
        self.train()
        self.likelihood.train()

        # --------------------------------------------------------- optimiser
        if reset_opt or self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:                                # keep momentum/state, just adjust LR
            for pg in self._optimizer.param_groups:
                pg["lr"] = lr
        opt = self._optimizer

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        iterator = tqdm(range(training_iterations), desc="Training", leave=False)

        for _ in iterator:
            opt.zero_grad()
            output = self(self.train_x_scaled)
            loss = -mll(output, self.train_y)
            loss.backward()
            opt.step()

            self._store_constrained_params(loss)
            iterator.set_postfix(loss=loss.item(),
                                 noise=self.likelihood.noise.item())

        # one-shot diagnostics
        self._bic   = self.compute_bic()
        self._loocv = self.get_loocv()
        if verbose:
            self.summary()
        return self

    # ------------------------------ fast data update ------------------------
    def set_train_data(self, *, inputs, targets, strict: bool = False):
        """
        Wrapper around gpytorch.models.ExactGP.set_train_data that also keeps
        the feature scaler in-sync.  Intended for Bayesian-optimisation loops.
        """
        inputs_t  = to_torch(inputs)
        targets_t = to_torch(targets)

        self.scaler.fit(inputs_t)
        inputs_s  = self.scaler.scale(inputs_t)

        super().set_train_data(inputs=inputs_s, targets=targets_t, strict=strict)

        # book-keeping mirrors
        self.train_x        = inputs_t
        self.train_y        = targets_t
        self.train_x_scaled = inputs_s


    # -------------------------------- prediction ---------------------------
    def make_predictions(
        self,
        test_x,
        *,
        type: str = "f",  # 'f' = latent, 'y' = noisy observations
        return_type: str = "numpy",
        posterior: bool = True,
    ):
        if posterior:
            self.eval()
            self.likelihood.eval()
        else:
            self.train()
            self.likelihood.train()

        test_x = to_torch(test_x)
        test_x_scaled = self.scaler.scale(test_x)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if type == "f":
                preds = self(test_x_scaled)
            elif type == "y":
                preds = self.likelihood(self(test_x_scaled))
            else:
                raise ValueError("type must be 'f' or 'y'.")

        predictions = Predictions(preds.mean, preds.variance)
        if return_type == "numpy":
            predictions.to_numpy()
        return predictions

    # -------------------------------- simulation ---------------------------
    def simulate(
        self,
        x_sim,
        *,
        method: str = "prior",  # 'prior' | 'posterior'
        type: str = "f",  # 'f' | 'y'
        return_type: str = "numpy",
        n_paths: int = 1,
    ):
        if n_paths > 1:
            raise NotImplementedError("call simulate repeatedly for multiple paths.")
        if method not in {"prior", "posterior"}:
            raise ValueError("method must be 'prior' or 'posterior'.")
        if type not in {"f", "y"}:
            raise ValueError("type must be 'f' or 'y'.")

        x_sim = to_torch(x_sim)
        x_scaled = self.scaler.scale(x_sim)

        # Switch GP mode
        if method == "prior":
            self.train()
        else:
            self.eval()

        with torch.no_grad():
            base_dist = self(x_scaled)
            if type == "f":
                samples = base_dist.rsample()
            else:  # 'y'
                samples = self.likelihood(base_dist).rsample()

        if return_type == "numpy":
            samples = to_numpy(samples)
        return samples

    # -------------------------------- diagnostics --------------------------
    def get_loocv(self):
        """Leave‑one‑out‐CV RMSE (latent *f* scale)."""
        self.train()
        y_dist = self.likelihood(self(self.train_x_scaled))
        K = y_dist.covariance_matrix
        y = self.train_y

        K_inv = torch.inverse(K)
        y_hat = y - (K_inv @ y.unsqueeze(-1)).squeeze() / torch.diag(K_inv)
        rmse = (y - y_hat).pow(2).mean().sqrt()
        return rmse.item()

    def compute_bic(self, data=None):
        self.train()
        X = self.train_x_scaled if data is None else to_torch(data)
        n = X.shape[0]

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self).to(X.device)
        log_marg_like = mll(self(X), self.train_y)

        self.num_param = sum(p.numel() for p in self.parameters())
        with torch.no_grad():
            bic = -log_marg_like * n + self.num_param * np.log(n) / 2
        return bic.item()

    # ---------------------------- reporting helpers ------------------------
    @property
    def bic(self):
        """Bayesian Information Criterion (cached after fitting)."""
        if self._bic is None:
            self._bic = self.compute_bic()
        return self._bic

    @property
    def loocv(self):
        """Leave‑one‑out RMSE (cached after fitting)."""
        if self._loocv is None:
            self._loocv = self.get_loocv()
        return self._loocv

    def summary(self):
        """Print high‑level model diagnostics."""
        print("GP model summary")
        print("----------------")
        print(f"Kernel          : {self.kernel}")
        print(f"# parameters    : {self.num_param}")
        print(f"BIC             : {self.bic:.3f}")
        print(f"LOOCV RMSE      : {self.loocv:.3f}\n")

        print("Fitted hyper‑parameters (constrained scale):")
        for name, param in self.named_hyperparameters():
            constraint = self.constraint_for_parameter_name(name)
            value = constraint.transform(param) if constraint is not None else param
            print(f"  {strip_raw_prefix(name):35s}: {to_numpy(value).ravel()}")
        print("----------------\n")

    # ------------------------------ plotting -------------------------------
    def plot_hyperparameters(self, ncols: int = 3, figsize_per_row: tuple[int, int] = (12, 3)):
        """Plot optimisation traces stored in ``param_history``."""
        param_names = [p for p in sorted(self.param_history) if p != "loss"]
        n_params = len(param_names) + 1
        nrows = math.ceil(n_params / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_row[0], figsize_per_row[1] * nrows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, pname in enumerate(param_names):
            ax = axes[i]
            vals = self.param_history[pname]

            # Scalars vs vectors
            if isinstance(vals[0], (float, int)) or np.isscalar(vals[0]):
                y = vals
            else:  # array‑valued hyper‑parameter → plot Euclidean norm
                y = [float(np.linalg.norm(v)) for v in vals]
            ax.plot(y)
            ax.set_title(pname)
            ax.set_xlabel("iteration")
            ax.grid(True, alpha=0.3)

        # Loss in a dedicated pane (if space) or last unused axis
        if len(axes) > n_params:
            ax_loss = axes[n_params]
            ax_loss.plot(self.param_history["loss"])
            ax_loss.set_title("negative MLL loss")
            ax_loss.set_xlabel("iteration")
            ax_loss.grid(True, alpha=0.3)

        # Hide unused axes
        for ax in axes[n_params + 1:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Auxiliary containers / utils
# -----------------------------------------------------------------------------
class Predictions:
    """Lightweight container for predictive mean and variance."""

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    # Mutating conversions for convenience ------------------------------------------------
    def to_numpy(self):
        self.mean = to_numpy(self.mean)
        self.variance = to_numpy(self.variance)

    def to_torch(self):
        self.mean = to_torch(self.mean)
        self.variance = to_torch(self.variance)


def strip_raw_prefix(name: str) -> str:
    """Remove gpytorch's "raw_" prefix from parameter names for readability."""
    parts = name.split(".")
    if parts[-1].startswith("raw_"):
        parts[-1] = parts[-1][4:]
    return ".".join(parts)
