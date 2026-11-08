"""Validate GPU acceleration for the causal-gpu environment. Exits 1 on failure."""
import json
import subprocess
import sys
import time
import warnings

import numpy as np
import torch

TRUE_ATE = 1.0
failures = []


def check(name, passed, detail=""):
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}")
    if not passed:
        failures.append(name)


def simulate(n, p, device):
    """Confounded observational data with a heterogeneous effect, tau(x) = 1 + 0.5*x1."""
    gen = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(n, p, device=device, dtype=torch.float64, generator=gen)
    propensity = torch.sigmoid(x[:, 0])  # x0 drives treatment and outcome
    t = (torch.rand(n, device=device, dtype=torch.float64, generator=gen) < propensity).double()
    noise = torch.randn(n, device=device, dtype=torch.float64, generator=gen)
    y = 2.0 * x[:, 0] - x[:, 2] + (TRUE_ATE + 0.5 * x[:, 1]) * t + noise
    return x, t, y


def interacted_ols_ate(x, t, y):
    """Estimate the ATE by solving the normal equations for y ~ 1 + X + T + T:X."""
    d = torch.cat([torch.ones_like(t)[:, None], x, t[:, None], t[:, None] * x], dim=1)
    beta = torch.linalg.solve(d.T @ d, d.T @ y)
    p = x.shape[1]
    return (beta[p + 1] + x.mean(0) @ beta[p + 2:]).item()


def timed_run(device, n=1_000_000, p=50):
    start = time.perf_counter()
    ate = interacted_ols_ate(*simulate(n, p, device))
    if device == "cuda":
        torch.cuda.synchronize()
    return ate, time.perf_counter() - start


print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__}")
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip())

check("CUDA build of PyTorch", torch.version.cuda is not None, f"CUDA runtime {torch.version.cuda}")
if not torch.cuda.is_available():
    check("torch.cuda.is_available", False)
    sys.exit(1)

props = torch.cuda.get_device_properties(0)
print(f"GPU 0: {props.name}, {props.total_memory / 2**30:.1f} GiB, CUDNN {torch.backends.cudnn.version()}")

# 1. Causal matrix operation on the GPU.
torch.cuda.reset_peak_memory_stats()
ate_gpu, t_gpu = timed_run("cuda")
peak = torch.cuda.max_memory_allocated() / 2**30
ate_cpu, t_cpu = timed_run("cpu")
check("GPU memory used by matrix solve", peak > 0.1, f"peak {peak:.2f} GiB")
check("Interacted OLS recovers ATE", abs(ate_gpu - TRUE_ATE) < 0.02, f"estimate {ate_gpu:.4f}, truth {TRUE_ATE}")
check("GPU and CPU estimates agree", abs(ate_gpu - ate_cpu) < 0.02, f"CPU estimate {ate_cpu:.4f}")
print(f"       Time: GPU {t_gpu:.2f}s, CPU {t_cpu:.2f}s, speedup {t_cpu / t_gpu:.1f}x")

# 2. Uplift libraries, using a smaller NumPy sample.
x, t, y = (a.cpu().numpy() for a in simulate(100_000, 10, "cuda"))
torch.cuda.empty_cache()

import xgboost as xgb
from causalml.inference.meta import BaseTRegressor
from econml.dml import LinearDML
from sklearn.linear_model import LassoCV, LogisticRegressionCV

check("XGBoost built with CUDA", xgb.build_info()["USE_CUDA"])
learner = xgb.XGBRegressor(device="cuda", tree_method="hist", n_estimators=200, max_depth=5)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # NumPy prediction input triggers a device-mismatch notice
    uplift = BaseTRegressor(learner=learner)
    ate, _, _ = uplift.estimate_ate(x, t.astype(int), y)
fitted = next(iter(uplift.models_t.values()))  # CausalML trains deep copies of the learner
config = json.loads(fitted.get_booster().save_config())
device = config["learner"]["generic_param"]["device"]
check("XGBoost training device", device.startswith("cuda"), device)
check("CausalML T-learner recovers ATE", abs(float(ate[0]) - TRUE_ATE) < 0.1, f"estimate {float(ate[0]):.4f}")

n = 10_000  # EconML DML is CPU-only; this confirms the compiled stack works
dml = LinearDML(model_y=LassoCV(), model_t=LogisticRegressionCV(), discrete_treatment=True, cv=2, random_state=0)
dml.fit(y[:n], t[:n].astype(int), X=x[:n, :3], W=x[:n, 3:])
ate = float(dml.ate(x[:n, :3]))
check("EconML LinearDML recovers ATE", abs(ate - TRUE_ATE) < 0.15, f"estimate {ate:.4f}")

print("\nRESULT:", f"FAILED: {failures}" if failures else "All checks passed.")
sys.exit(1 if failures else 0)
