"""Hotel stay-extension uplift pipeline on the Kevin Hillstrom (MineThatData) email RCT.

Framing: a hotel e-mails a promotion to past guests. Treatment T = any e-mail
(Mens or Womens campaign) vs. no e-mail. Outcome Y = incremental nights booked,
derived from the continuous `spend` column (spend / NIGHT_SPEND).

Models (all fitted on the RTX 3090 unless noted)
    XGBoost (device="cuda") ........ S/T/X/Domain-Adaptation, DR, Linear-DR, Linear-DML,
                                     NonParam-DML, Causal Forest and Forest-DR nuisances,
                                     CausalML S/T/X/R learners
    PyTorch (cuda) ................. TARNet, DragonNet (custom neural CATE models)
    CPU only ....................... forest final stages (EconML CausalForestDML, ForestDRLearner)
                                     - no GPU implementation exists in EconML

Outputs
    figures/   EDA and model diagnostic plots (PNG)
    models/    fitted models (joblib; PyTorch state dicts as .pt)
    results/   validation.json, summary.csv, decile_<model>.csv
"""

import json
import time
import traceback
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import xgboost as xgb  # noqa: E402
from causalml.inference.meta import BaseRRegressor, BaseSRegressor, BaseTRegressor, BaseXRegressor  # noqa: E402
from econml.dml import CausalForestDML, LinearDML, NonParamDML  # noqa: E402
from econml.dr import DRLearner, ForestDRLearner, LinearDRLearner  # noqa: E402
from econml.metalearners import DomainAdaptationLearner, SLearner, TLearner, XLearner  # noqa: E402
from sklift.datasets import fetch_hillstrom  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
# XGBoost warns on every numpy predict on a CUDA booster; the result is still correct.
warnings.filterwarnings("ignore", category=UserWarning, message=".*mismatched devices.*")
warnings.filterwarnings("ignore", message=".*Potential solutions.*")

ROOT = Path(__file__).resolve().parent
FIG, MODELS, RES = ROOT / "figures", ROOT / "models", ROOT / "results"
for _d in (FIG, MODELS, RES):
    _d.mkdir(exist_ok=True)

SEED = 42
ADR = 180.0            # average daily rate, $ per night
OFFER_COST = 15.0      # promotional offer cost, $ per customer e-mailed
NIGHT_SPEND = 150.0    # spend $ that corresponds to one incremental night

NUMERIC = {
    "recency": "months_since_last_stay",
    "history": "historical_room_spend",
    "mens": "leisure_package_affinity",
    "womens": "family_package_affinity",
    "newbie": "first_stay_flag",
}
CATEGORICAL = {"zip_code": "drive_market_tier", "channel": "booking_channel", "history_segment": "spend_band"}


# ------------------------------------------------------------------------- GPU setup
def assert_gpu():
    """Fail fast unless CUDA is visible to both PyTorch and XGBoost."""
    assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
    probe = xgb.XGBRegressor(device="cuda", tree_method="hist", n_estimators=5)
    probe.fit(np.random.rand(200, 3), np.random.rand(200))
    dev = json.loads(probe.get_booster().save_config())["learner"]["generic_param"]["device"]
    assert dev.startswith("cuda"), f"XGBoost device is {dev}"
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | XGBoost device {dev}")
    return dev


# ----------------------------------------------------------------------------- data
def load() -> pd.DataFrame:
    X, spend, segment = fetch_hillstrom(target_col="spend", return_X_y_t=True)
    df = X.rename(columns={**NUMERIC, **CATEGORICAL}).reset_index(drop=True)
    df["spend"] = spend.to_numpy()
    df["segment"] = segment.to_numpy()
    df["T"] = (df["segment"] != "No E-Mail").astype(int)
    df["nights"] = df["spend"] / NIGHT_SPEND
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    num = df[list(NUMERIC.values())].astype(float)
    cat = pd.get_dummies(df[list(CATEGORICAL.values())], drop_first=True, dtype=float)
    return pd.concat([num, cat], axis=1)


# ------------------------------------------------------------- GPU nuisance learners
def xr():
    return xgb.XGBRegressor(device="cuda", tree_method="hist", n_estimators=400, learning_rate=0.05,
                            max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=SEED)


def xc():
    return xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=300, learning_rate=0.05,
                             max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=SEED)


# --------------------------------------------------------------------- PyTorch CATE
class TwoHeadNet(nn.Module):
    """Shared representation with outcome heads for control and treated (TARNet / DragonNet)."""

    def __init__(self, d: int, h: int = 128, dragon: bool = False):
        super().__init__()
        self.rep = nn.Sequential(nn.Linear(d, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, h), nn.ELU())
        self.y0 = nn.Sequential(nn.Linear(h, h), nn.ELU(), nn.Linear(h, 1))
        self.y1 = nn.Sequential(nn.Linear(h, h), nn.ELU(), nn.Linear(h, 1))
        self.dragon = dragon
        if dragon:
            self.prop = nn.Sequential(nn.Linear(h, h), nn.ELU(), nn.Linear(h, 1))
            self.eps = nn.Parameter(torch.zeros(1))  # targeted-regularisation perturbation

    def forward(self, x):
        r = self.rep(x)
        logit = self.prop(r).squeeze(-1) if self.dragon else None
        return self.y0(r).squeeze(-1), self.y1(r).squeeze(-1), logit


def train_neural(Xtr, Ttr, Ytr, dragon: bool, epochs: int = 40, bs: int = 512, lr: float = 1e-3):
    dev = torch.device("cuda")
    torch.manual_seed(SEED)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = float(Ytr.mean()), float(Ytr.std())
    X = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    T = torch.tensor(Ttr, dtype=torch.float32, device=dev)
    Y = torch.tensor((Ytr - ym) / ys, dtype=torch.float32, device=dev)
    net = TwoHeadNet(X.shape[1], dragon=dragon).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.functional.mse_loss
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            y0, y1, logit = net(X[b])
            yhat = torch.where(T[b] > 0.5, y1, y0)
            loss = mse(yhat, Y[b])
            if dragon:
                g = torch.sigmoid(logit).clamp(0.01, 0.99)
                loss = loss + nn.functional.binary_cross_entropy_with_logits(logit, T[b])
                h = T[b] / g - (1 - T[b]) / (1 - g)
                loss = loss + mse(yhat + net.eps * h, Y[b])  # targeted regularisation
            opt.zero_grad()
            loss.backward()
            opt.step()
    bundle = {"state": {k: v.cpu() for k, v in net.state_dict().items()}, "mu": mu, "sd": sd,
              "ym": ym, "ys": ys, "d": X.shape[1], "dragon": dragon}
    return bundle


def predict_neural(bundle, Xte) -> np.ndarray:
    dev = torch.device("cuda")
    net = TwoHeadNet(bundle["d"], dragon=bundle["dragon"]).to(dev)
    net.load_state_dict(bundle["state"])
    net.eval()
    X = torch.tensor((Xte - bundle["mu"]) / bundle["sd"], dtype=torch.float32, device=dev)
    with torch.no_grad():
        y0, y1, _ = net(X)
    return ((y1 - y0) * bundle["ys"]).cpu().numpy()  # CATE in nights


# ------------------------------------------------------------------ model registry
def econ(factory):
    def run(Xtr, Ttr, Ytr, Xte):
        est = factory()
        est.fit(Ytr, Ttr, X=Xtr)
        return np.ravel(est.effect(Xte)), est
    return run


def causalml(cls):
    def run(Xtr, Ttr, Ytr, Xte):
        m = cls(learner=xr(), control_name=0)
        m.fit(X=Xtr, treatment=Ttr, y=Ytr)
        return np.ravel(m.predict(X=Xte)), m
    return run


def neural(dragon: bool):
    def run(Xtr, Ttr, Ytr, Xte):
        bundle = train_neural(Xtr, Ttr, Ytr, dragon=dragon)
        return predict_neural(bundle, Xte), bundle
    return run


MODEL_SPECS = [
    # EconML meta-learners (XGBoost on GPU)
    {"name": "S-Learner", "family": "EconML meta", "device": "cuda (XGBoost)",
     "run": econ(lambda: SLearner(overall_model=xr()))},
    {"name": "T-Learner", "family": "EconML meta", "device": "cuda (XGBoost)",
     "run": econ(lambda: TLearner(models=xr()))},
    {"name": "X-Learner", "family": "EconML meta", "device": "cuda (XGBoost)",
     "run": econ(lambda: XLearner(models=xr(), cate_models=xr(), propensity_model=xc()))},
    {"name": "Domain-Adaptation", "family": "EconML meta", "device": "cuda (XGBoost)",
     "run": econ(lambda: DomainAdaptationLearner(models=xr(), final_models=xr(), propensity_model=xc()))},
    # EconML doubly-robust learners
    {"name": "DRLearner", "family": "EconML DR", "device": "cuda (XGBoost)",
     "run": econ(lambda: DRLearner(model_regression=xr(), model_propensity=xc(), model_final=xr(), cv=3,
                                   random_state=SEED))},
    {"name": "LinearDRLearner", "family": "EconML DR", "device": "cuda (XGBoost nuisances)",
     "run": econ(lambda: LinearDRLearner(model_regression=xr(), model_propensity=xc(), cv=3,
                                         random_state=SEED))},
    {"name": "ForestDRLearner", "family": "EconML DR", "device": "cuda nuisances + CPU forest",
     "run": econ(lambda: ForestDRLearner(model_regression=xr(), model_propensity=xc(), cv=3, n_estimators=200,
                                         random_state=SEED, n_jobs=-1))},
    # EconML double machine learning
    {"name": "LinearDML", "family": "EconML DML", "device": "cuda (XGBoost nuisances)",
     "run": econ(lambda: LinearDML(model_y=xr(), model_t=xc(), discrete_treatment=True, cv=3,
                                   random_state=SEED))},
    {"name": "NonParamDML", "family": "EconML DML", "device": "cuda (XGBoost)",
     "run": econ(lambda: NonParamDML(model_y=xr(), model_t=xc(), model_final=xr(), discrete_treatment=True,
                                     cv=3, random_state=SEED))},
    {"name": "CausalForestDML", "family": "EconML forest", "device": "cuda nuisances + CPU forest",
     "run": econ(lambda: CausalForestDML(model_y=xr(), model_t=xc(), discrete_treatment=True, n_estimators=200,
                                         min_samples_leaf=20, cv=3, random_state=SEED, n_jobs=-1))},
    # CausalML meta-learners (XGBoost on GPU)
    {"name": "CausalML-S", "family": "CausalML meta", "device": "cuda (XGBoost)", "run": causalml(BaseSRegressor)},
    {"name": "CausalML-T", "family": "CausalML meta", "device": "cuda (XGBoost)", "run": causalml(BaseTRegressor)},
    {"name": "CausalML-X", "family": "CausalML meta", "device": "cuda (XGBoost)", "run": causalml(BaseXRegressor)},
    {"name": "CausalML-R", "family": "CausalML meta", "device": "cuda (XGBoost)", "run": causalml(BaseRRegressor)},
    # PyTorch neural CATE models (CUDA)
    {"name": "TARNet", "family": "Neural", "device": "cuda (PyTorch)", "run": neural(dragon=False)},
    {"name": "DragonNet", "family": "Neural", "device": "cuda (PyTorch)", "run": neural(dragon=True)},
]


# ----------------------------------------------------------------------- statistics
def diff_ci(y, t):
    """Difference in means treated - control with a 95% normal CI."""
    y1, y0 = y[t == 1], y[t == 0]
    d = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return float(d), float(d - 1.96 * se), float(d + 1.96 * se)


def qini_curve(pred, y, t):
    """Cumulative incremental-gain curve when targeting customers in descending predicted CATE."""
    o = np.argsort(-pred)
    y, t = y[o], t[o]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    y_t, y_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    scale = np.divide(n_t, n_c, out=np.zeros_like(n_t, dtype=float), where=n_c > 0)
    q = y_t - y_c * scale
    frac = np.arange(1, len(y) + 1) / len(y)
    return np.r_[0.0, frac], np.r_[0.0, q]


def decile_table(pred, y, t) -> pd.DataFrame:
    d = pd.DataFrame({"pred": pred, "y": y, "t": t})
    d["decile"] = pd.qcut(d["pred"].rank(method="first", ascending=False), 10, labels=np.arange(1, 11)).astype(int)
    rows = []
    for k, g in d.groupby("decile"):
        g1, g0 = g.loc[g.t == 1, "y"], g.loc[g.t == 0, "y"]
        lift = g1.mean() - g0.mean()
        se = np.sqrt(g1.var(ddof=1) / len(g1) + g0.var(ddof=1) / len(g0))
        pred_rev = g["pred"].mean() * ADR - OFFER_COST
        act_rev = lift * ADR - OFFER_COST
        rows.append({
            "decile": k,
            "n": len(g),
            "n_treated": len(g1),
            "n_control": len(g0),
            "pred_lift_nights": g["pred"].mean(),
            "actual_lift_nights": lift,
            "actual_se_nights": se,
            "pred_rev_per_cust": pred_rev,
            "actual_rev_per_cust": act_rev,
            "predicted_net_revenue": pred_rev * len(g),
            "actual_net_revenue": act_rev * len(g),
        })
    out = pd.DataFrame(rows)
    out["cum_actual_net_revenue"] = out["actual_net_revenue"].cumsum()
    return out


def best_cutoff(tab: pd.DataFrame):
    """Number of top deciles to target that maximises cumulative actual net revenue."""
    cum = tab["actual_net_revenue"].cumsum().to_numpy()
    k = int(np.argmax(cum)) + 1
    return k, float(cum[k - 1])


def slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------- plots
def plot_eda(df: pd.DataFrame, F: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    order = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]
    cap = df["spend"].quantile(0.99)
    data = [df.loc[df.segment == s, "spend"].clip(upper=cap) for s in order]
    ax[0].boxplot(data, tick_labels=order, showfliers=False)
    ax[0].set_title("Spend by arm (clipped at 99th pct)")
    ax[0].set_ylabel("Spend ($)")
    means = df.groupby("segment")["spend"].mean().reindex(order)
    se = df.groupby("segment")["spend"].sem().reindex(order)
    ax[1].bar(order, means, yerr=1.96 * se, capsize=6, color=["#888", "#3b6ea8", "#c2567a"])
    ax[1].set_title("Mean spend per customer with 95% CI")
    ax[1].set_ylabel("Mean spend ($)")
    fig.suptitle("Email RCT: outcome by arm")
    fig.tight_layout()
    fig.savefig(FIG / "eda_outcome_by_arm.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax_, col in zip(axes.ravel(), NUMERIC.values()):
        ax_.hist(df.loc[df["T"] == 0, col], bins=30, alpha=0.5, density=True, label="Control")
        ax_.hist(df.loc[df["T"] == 1, col], bins=30, alpha=0.5, density=True, label="E-mailed")
        ax_.set_title(col)
        ax_.legend()
    axes.ravel()[-1].axis("off")
    fig.suptitle("Guest profile distributions by treatment")
    fig.tight_layout()
    fig.savefig(FIG / "eda_feature_distributions.png", dpi=130)
    plt.close(fig)

    T = df["T"].to_numpy()
    a, b = F[T == 1], F[T == 0]
    smd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    smd = smd.replace([np.inf, -np.inf], 0).fillna(0).sort_values()
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(smd.index, smd.values, color=np.where(np.abs(smd) > 0.1, "#c2567a", "#3b6ea8"))
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(0.1, ls="--", color="grey")
    ax.axvline(-0.1, ls="--", color="grey")
    ax.set_title("Covariate balance: standardised mean difference (E-mailed - control)")
    fig.tight_layout()
    fig.savefig(FIG / "eda_covariate_balance.png", dpi=130)
    plt.close(fig)


def plot_cate_all(cates: dict):
    fig, ax = plt.subplots(figsize=(13, 6))
    names = list(cates)
    ax.boxplot([cates[n] for n in names], tick_labels=names, showfliers=False)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Predicted incremental nights per guest")
    ax.set_title("Individual uplift (CATE) distribution by model, hold-out guests")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "cate_distribution_by_model.png", dpi=130)
    plt.close(fig)


def plot_cate_detail(p: np.ndarray, Xte_df: pd.DataFrame, seg_te: np.ndarray, tier_te: np.ndarray, cf, label: str):
    idx = np.random.default_rng(SEED).choice(len(p), size=min(4000, len(p)), replace=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    for s, c in [("Mens E-Mail", "#3b6ea8"), ("Womens E-Mail", "#c2567a")]:
        m = idx[seg_te[idx] == s]
        ax.scatter(Xte_df["historical_room_spend"].to_numpy()[m], p[m], s=6, alpha=0.4, c=c, label=s)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Historical room spend ($)")
    ax.set_ylabel(f"{label} CATE (nights)")
    ax.set_title(f"Who responds to the stay-extension e-mail? ({label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "cate_vs_history_causal_forest.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].boxplot([p[seg_te == s] for s in ["Mens E-Mail", "Womens E-Mail"]],
                  tick_labels=["Mens E-Mail", "Womens E-Mail"], showfliers=False)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title(f"{label} CATE by campaign")
    ax[0].set_ylabel("Incremental nights")
    tiers = sorted(set(tier_te))
    ax[1].boxplot([p[tier_te == t] for t in tiers], tick_labels=tiers, showfliers=False)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title(f"{label} CATE by drive-market tier")
    fig.tight_layout()
    fig.savefig(FIG / "cate_by_segment_and_tier.png", dpi=130)
    plt.close(fig)

    if cf is not None:
        try:
            imp = pd.Series(np.ravel(cf.feature_importances_), index=list(Xte_df.columns)).sort_values()
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.barh(imp.index, imp.values, color="#3b6ea8")
            ax.set_title("Causal Forest feature importance (heterogeneity drivers)")
            fig.tight_layout()
            fig.savefig(FIG / "causal_forest_feature_importance.png", dpi=130)
            plt.close(fig)
        except Exception as exc:
            print(f"  [warn] feature importance plot skipped: {exc}")


def plot_qini(curves: dict):
    fig, ax = plt.subplots(figsize=(11, 7))
    for name, (x, y) in curves.items():
        ax.plot(x, y, lw=1.6, label=name)
    x_any, y_any = next(iter(curves.values()))
    ax.plot([0, 1], [0, y_any[-1]], "k--", lw=2, label="Random targeting")
    ax.set_xlabel("Fraction of guests targeted (highest predicted uplift first)")
    ax.set_ylabel("Cumulative incremental spend gain ($)")
    ax.set_title("Qini curves on hold-out guests, all models")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(FIG / "qini_curves.png", dpi=130)
    plt.close(fig)


def plot_decile_single(name: str, tab: pd.DataFrame):
    """One predicted-vs-actual decile chart per model, written to results/decile_charts/."""
    out_dir = RES / "decile_charts"
    out_dir.mkdir(exist_ok=True)
    x = tab["decile"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, tab["pred_lift_nights"], width=0.4, label="Predicted lift")
    ax.bar(x + 0.2, tab["actual_lift_nights"], width=0.4, yerr=1.96 * tab["actual_se_nights"],
           capsize=3, label="Actual lift (RCT, 95% CI)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("Decile (1 = highest predicted uplift)")
    ax.set_ylabel("Incremental nights per guest")
    ax.set_title(f"{name}: predicted vs actual uplift by decile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug(name)}.png", dpi=130)
    plt.close(fig)


def plot_deciles(tables: dict):
    n = len(tables)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, tab) in zip(axes, tables.items()):
        x = tab["decile"].to_numpy()
        ax.bar(x - 0.2, tab["pred_lift_nights"], width=0.4, label="Predicted")
        ax.bar(x + 0.2, tab["actual_lift_nights"], width=0.4, yerr=1.96 * tab["actual_se_nights"],
               capsize=2, label="Actual (RCT)")
        ax.set_title(name, fontsize=10)
        ax.set_xticks(x)
        ax.axhline(0, color="k", lw=0.8)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend()
    fig.supxlabel("Decile (1 = highest predicted uplift)")
    fig.supylabel("Incremental nights per guest")
    fig.suptitle("Predicted vs actual uplift by decile")
    fig.tight_layout()
    fig.savefig(FIG / "decile_predicted_vs_actual.png", dpi=130)
    fig.savefig(RES / "decile_predicted_vs_actual.png", dpi=130)
    plt.close(fig)


def plot_revenue_heatmap(tables: dict):
    mat = pd.DataFrame({name: tab.set_index("decile")["actual_net_revenue"] for name, tab in tables.items()}).T
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(mat) + 2))
    vmax = np.abs(mat.to_numpy()).max()
    im = ax.imshow(mat.to_numpy(), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(10), labels=[str(i) for i in range(1, 11)])
    ax.set_yticks(range(len(mat)), labels=mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iat[i, j] / 1000:.1f}k", ha="center", va="center", fontsize=7)
    ax.set_xlabel("Decile (1 = highest predicted uplift)")
    ax.set_title(f"Actual net incremental revenue per decile ($; ADR ${ADR:.0f}, offer ${OFFER_COST:.0f})")
    fig.colorbar(im, ax=ax, label="$")
    fig.tight_layout()
    fig.savefig(FIG / "net_revenue_heatmap.png", dpi=130)
    fig.savefig(RES / "net_revenue_heatmap.png", dpi=130)
    plt.close(fig)


def plot_model_comparison(summary_df: pd.DataFrame):
    df = summary_df.sort_values("qini_gain_over_random")
    colors = ["#3b6ea8" if "cuda" in d and "CPU" not in d else "#c2567a" for d in df["device"]]
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(df) + 2))
    ax.barh(df["model"], df["qini_gain_over_random"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Qini gain over random targeting ($, higher is better)")
    ax.set_title("Model ranking (blue = fully GPU, red = CPU forest in the pipeline)")
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    dev = assert_gpu()
    df = load()
    F = features(df)
    T = df["T"].to_numpy()
    Y = df["nights"].to_numpy()
    seg = df["segment"].to_numpy()

    print(f"Loaded {len(df):,} rows, {F.shape[1]} features; treated share {T.mean():.3f}")
    plot_eda(df, F)

    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3, stratify=seg, random_state=SEED)
    Xtr, Xte = F.iloc[idx_tr].to_numpy(), F.iloc[idx_te].to_numpy()
    Ttr, Tte = T[idx_tr], T[idx_te]
    Ytr, Yte = Y[idx_tr], Y[idx_te]
    seg_te = seg[idx_te]
    Xte_df = F.iloc[idx_te].reset_index(drop=True)
    tier_te = df["drive_market_tier"].iloc[idx_te].to_numpy()

    naive_nights = diff_ci(Yte, Tte)
    print(f"Naive RCT ATE (test): {naive_nights[0]:.4f} nights [{naive_nights[1]:.4f}, {naive_nights[2]:.4f}]")

    # ---- fit every model
    fitted, errors, timing = {}, {}, {}
    for spec in MODEL_SPECS:
        name = spec["name"]
        print(f"\nFitting {name} [{spec['device']}] ...")
        t0 = time.perf_counter()
        try:
            pred, obj = spec["run"](Xtr, Ttr, Ytr, Xte)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {errors[name]}")
            traceback.print_exc()
            continue
        timing[name] = time.perf_counter() - t0
        fitted[name] = {"pred": pred, "obj": obj, "spec": spec}
        print(f"  done in {timing[name]:.1f}s, mean CATE {pred.mean():.4f} nights")

    # ---- placebo on the GPU T-Learner: permuted treatment should give ~0
    rng = np.random.default_rng(SEED)
    placebo_pred, _ = econ(lambda: TLearner(models=xr()))(Xtr, rng.permutation(Ttr), Ytr, Xte)
    placebo_ate = float(placebo_pred.mean())
    print(f"\nPlacebo ATE (permuted T, T-Learner): {placebo_ate:.4f} nights")

    # ---- tables, qini, summary
    tables, curves, summary = {}, {}, []
    for name, info in fitted.items():
        pred = info["pred"]
        tab = decile_table(pred, Yte, Tte)
        tab.to_csv(RES / f"decile_{slug(name)}.csv", index=False)
        tables[name] = tab
        x, q = qini_curve(pred, Yte, Tte)
        curves[name] = (x, q)
        auc = float(np.trapezoid(q, x))
        random_auc = float(0.5 * q[-1])
        k, net_best = best_cutoff(tab)
        corr = float(np.corrcoef(tab["pred_lift_nights"], tab["actual_lift_nights"])[0, 1])
        summary.append({
            "model": name,
            "family": info["spec"]["family"],
            "device": info["spec"]["device"],
            "fit_seconds": round(timing[name], 1),
            "mean_cate_nights": float(pred.mean()),
            "qini_auc": auc,
            "random_auc": random_auc,
            "qini_gain_over_random": auc - random_auc,
            "decile_pred_vs_actual_corr": corr,
            "net_revenue_all_customers": float(tab["actual_net_revenue"].sum()),
            "best_target_deciles": k,
            "net_revenue_best_cutoff": net_best,
        })

    summary_df = pd.DataFrame(summary).sort_values("qini_gain_over_random", ascending=False)
    summary_df.to_csv(RES / "summary.csv", index=False)
    print("\n" + summary_df.round(3).to_string(index=False))
    if errors:
        print("\nFailed models:", json.dumps(errors, indent=2))

    # ---- plots
    cates = {n: info["pred"] for n, info in fitted.items()}
    plot_cate_all(cates)
    cf_name = "CausalForestDML"
    if cf_name in fitted:
        plot_cate_detail(cates[cf_name], Xte_df, seg_te, tier_te, fitted[cf_name]["obj"], "Causal Forest")
    if curves:
        plot_qini(curves)
        plot_deciles(tables)
        plot_revenue_heatmap(tables)
        for name, tab in tables.items():
            plot_decile_single(name, tab)
        plot_model_comparison(summary_df)

    # ---- persist models
    saved = []
    for name, info in fitted.items():
        try:
            if info["spec"]["family"] == "Neural":
                torch.save(info["obj"], MODELS / f"{slug(name)}.pt")
            else:
                joblib.dump(info["obj"], MODELS / f"{slug(name)}.joblib")
            saved.append(name)
        except Exception as exc:
            print(f"  [warn] could not save {name}: {exc}")
    (MODELS / "feature_names.json").write_text(json.dumps(list(F.columns), indent=2))

    # ---- validation report
    seg_effects = {}
    for s in ["Mens E-Mail", "Womens E-Mail"]:
        mask = (seg == s) | (seg == "No E-Mail")
        t_s = (seg[mask] == s).astype(int)
        seg_effects[s] = diff_ci(Y[mask], t_s)[0] * NIGHT_SPEND  # $ spend per customer

    cf_ci = None
    if cf_name in fitted:
        try:
            inf = fitted[cf_name]["obj"].ate_inference(X=Xte)
            cf_ci = [float(v) for v in np.ravel(inf.conf_int_mean())]
        except Exception as exc:
            print(f"  [warn] causal forest ATE CI unavailable: {exc}")

    validation = {
        "gpu": {"device": torch.cuda.get_device_name(0), "xgboost_device": dev},
        "data": {
            "source": "sklift.datasets.fetch_hillstrom (Kevin Hillstrom MineThatData email RCT)",
            "rows": int(len(df)),
            "train_rows": int(len(idx_tr)),
            "test_rows": int(len(idx_te)),
            "treated_share": float(T.mean()),
        },
        "assumptions": {
            "night_spend_dollars": NIGHT_SPEND,
            "adr_dollars": ADR,
            "offer_cost_dollars_per_customer": OFFER_COST,
            "outcome": "incremental nights = spend / night_spend (continuous)",
        },
        "naive_rct_ate_test": {"nights": naive_nights, "spend_dollars": [v * NIGHT_SPEND for v in naive_nights]},
        "placebo_ate_permuted_treatment_nights": placebo_ate,
        "segment_effect_spend_dollars": seg_effects,
        "causal_forest_ate_ci_95_nights": cf_ci,
        "models_fitted": list(fitted),
        "models_failed": errors,
        "models_saved": saved,
        "models": summary,
        "decile_tables": {k: v.round(6).to_dict(orient="records") for k, v in tables.items()},
    }
    (RES / "validation.json").write_text(json.dumps(validation, indent=2, default=float))
    print(f"\nFitted {len(fitted)}/{len(MODEL_SPECS)} models. Saved figures to {FIG}, models to {MODELS}, "
          f"results to {RES}")


if __name__ == "__main__":
    main()
