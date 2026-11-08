"""Scenario A: loyalty tier upgrade offer, binary treatment with selection bias.

Quasi-experimental setting
    Hotel revenue management offers a loyalty tier upgrade to guests it judges
    valuable: frequent stayers, high past spend, business travellers and late
    bookers. Because the offer is not randomised, a naive comparison of
    upgraded vs. non-upgraded guests is biased upward.

Estimators
    LinearDML          Main model. Double machine learning with XGBoost nuisance
                       models (GPU if available). The final stage is linear in the
                       effect modifiers, so coefficients are an interpretable
                       heterogeneity table.
    CausalForestDML    Robustness check. Non-parametric CATE with feature importance.

Validation
    * Naive vs. DML vs. true ATE (the simulation knows every guest's true effect).
    * Overlap (propensity) diagnostics.
    * Out-of-sample CATE accuracy against the known true effect.
    * Refutation: placebo treatment and a random common cause.
    * Confidence-interval coverage of the true ATE.

Outputs
    figures/   diagnostic and result charts (PNG)
    results/   validation JSON, CSV tables
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from econml.dml import CausalForestDML, LinearDML  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*mismatched devices.*")
warnings.filterwarnings("ignore", message=".*Potential solutions.*")

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

SEED = 42
N_GUESTS = 30_000
ALPHA = 0.05
PROPERTY_TYPES = ["City", "Resort", "Budget", "Boutique"]  # City is the reference level


# ----------------------------------------------------------------------------- device
def pick_device() -> str:
    """Use the GPU for XGBoost when a CUDA build and device are available, else CPU."""
    try:
        probe = xgb.XGBRegressor(device="cuda", n_estimators=2)
        probe.fit(np.random.rand(20, 2), np.random.rand(20))
        return "cuda"
    except Exception:
        return "cpu"


def xgb_regressor(device: str):
    return xgb.XGBRegressor(device=device, tree_method="hist", n_estimators=300, learning_rate=0.05,
                            max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=SEED)


def xgb_classifier(device: str):
    return xgb.XGBClassifier(device=device, tree_method="hist", n_estimators=300, learning_rate=0.05,
                             max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=SEED)


# ------------------------------------------------------------------ synthetic data
def simulate(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Hotel guests with confounded loyalty upgrade offers and a known heterogeneous effect."""
    # Guest history (confounders)
    prior_stays = rng.poisson(2.0, n)                                   # stays in last 24 months
    prior_spend = rng.lognormal(5.5 + 0.15 * prior_stays, 0.5, n)       # $ spend in last 12 months
    business_share = rng.beta(2, 3, n)                                  # share of business stays
    lead_time = rng.gamma(2.0, 15.0, n)                                 # days booked before arrival
    party_size = rng.choice([1, 2, 3, 4], n, p=[0.35, 0.45, 0.12, 0.08])
    segment = rng.choice(["OTA", "Direct", "Corporate"], n, p=[0.45, 0.35, 0.20])

    # Stay context (seasonality, calendar, local demand)
    month = rng.integers(1, 13, n)
    dow = rng.integers(0, 7, n)
    weekend = (dow >= 5).astype(int)
    local_event = rng.binomial(1, 0.12, n)
    weather = rng.normal(0, 1, n)
    prop = rng.choice(PROPERTY_TYPES, n, p=[0.40, 0.20, 0.25, 0.15])
    is_resort = (prop == "Resort").astype(int)
    is_budget = (prop == "Budget").astype(int)
    is_boutique = (prop == "Boutique").astype(int)

    # Selection into treatment: the hotel targets valuable, business, late-booking guests.
    logit = (-2.2 + 0.45 * prior_stays + 0.0012 * prior_spend - 0.01 * lead_time
             + 1.2 * business_share + 0.3 * (segment == "Corporate"))
    p_treat = 1 / (1 + np.exp(-logit))
    treat = rng.binomial(1, p_treat)

    # Outcome: 90-day spend ($). Baseline depends on confounders; the treatment effect
    # depends on property type and business share (effect modifiers).
    seasonal = 40 * np.sin(2 * np.pi * (month - 1) / 12)
    baseline = (350 + 60 * prior_stays + 0.6 * prior_spend + 180 * business_share - 1.5 * lead_time
                + 25 * party_size + 30 * weekend + 45 * local_event + 20 * weather + seasonal
                + 0.0 * dow + 35 * (segment == "Corporate") - 20 * (segment == "OTA"))
    tau = 40 + 90 * is_resort - 35 * is_budget + 15 * is_boutique + 60 * business_share + 5 * prior_stays
    spend = baseline + tau * treat + rng.normal(0, 120, n)

    return pd.DataFrame({
        "guest_id": np.arange(n),
        "prior_stays": prior_stays,
        "prior_spend": prior_spend,
        "business_share": business_share,
        "lead_time_days": lead_time,
        "party_size": party_size,
        "segment": segment,
        "month": month,
        "dow": dow,
        "weekend": weekend,
        "local_event": local_event,
        "weather": weather,
        "property_type": prop,
        "is_resort": is_resort,
        "is_budget": is_budget,
        "is_boutique": is_boutique,
        "upgrade_offer": treat,
        "p_true": p_treat,
        "tau_true": tau,
        "spend_90d": spend,
    })


def build_matrices(df: pd.DataFrame):
    """X = effect modifiers (heterogeneity), W = other confounders and controls."""
    # Effect modifiers: where the upgrade's value can differ across guests.
    X = pd.DataFrame({
        "prior_stays": df["prior_stays"],
        "business_share": df["business_share"],
        "is_resort": df["is_resort"],
        "is_budget": df["is_budget"],
        "is_boutique": df["is_boutique"],
    })
    # Controls: everything else that drives selection or spend.
    W = pd.concat([
        df[["prior_spend", "lead_time_days", "party_size", "weekend", "local_event", "weather"]],
        pd.DataFrame({
            "month_sin": np.sin(2 * np.pi * df["month"] / 12),
            "month_cos": np.cos(2 * np.pi * df["month"] / 12),
        }, index=df.index),
        pd.get_dummies(df["dow"], prefix="dow", drop_first=True, dtype=float),
        pd.get_dummies(df["segment"], prefix="seg", drop_first=True, dtype=float),
    ], axis=1)
    return X, W


# ------------------------------------------------------------------------ estimators
def naive_difference(df: pd.DataFrame) -> dict:
    """Unadjusted treated-minus-untreated mean, the biased benchmark."""
    t = df.loc[df["upgrade_offer"] == 1, "spend_90d"]
    c = df.loc[df["upgrade_offer"] == 0, "spend_90d"]
    d = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return {"ate": float(d), "ci_low": float(d - 1.96 * se), "ci_high": float(d + 1.96 * se)}


def fit_linear_dml(X, W, T, Y, device: str) -> LinearDML:
    """DML: cross-fit outcome and treatment models on controls, then regress residuals on X.

    Cross-fitting (cv=5) avoids overfitting bias in the nuisance models; the linear final stage
    gives valid standard errors for the average and heterogeneous effects.
    """
    est = LinearDML(
        model_y=xgb_regressor(device),       # E[Y | X, W] on GPU
        model_t=xgb_classifier(device),      # P(T = 1 | X, W) on GPU, gives propensity-based residuals
        discrete_treatment=True,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        random_state=SEED,
    )
    est.fit(Y, T, X=X, W=W)
    return est


def fit_causal_forest(X, W, T, Y, device: str) -> CausalForestDML:
    """Non-parametric CATE. Forest step runs on CPU (no GPU implementation in EconML)."""
    est = CausalForestDML(
        model_y=xgb_regressor(device),
        model_t=xgb_classifier(device),
        discrete_treatment=True,
        n_estimators=400,
        min_samples_leaf=40,
        cv=5,
        random_state=SEED,
        n_jobs=-1,
    )
    est.fit(Y, T, X=X, W=W)
    return est


# ------------------------------------------------------------------------ validation
def overlap_propensity(X, W, T, device: str) -> np.ndarray:
    """Cross-fitted propensity scores, used to check positivity (common support)."""
    Z = np.hstack([X.to_numpy(), W.to_numpy()])
    p = np.zeros(len(T))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in cv.split(Z, T):
        clf = xgb_classifier(device).fit(Z[tr], T[tr])
        p[te] = clf.predict_proba(Z[te])[:, 1]
    return p


def ci_dict(point, low, high) -> dict:
    return {"estimate": float(point), "ci_low": float(low), "ci_high": float(high),
            "covers_zero": bool(low <= 0 <= high)}


def segment_ate(est, X: pd.DataFrame, mask: np.ndarray):
    """ATE over a subpopulation: the average marginal effect for rows in `mask`."""
    point = est.ate(X=X[mask])
    low, high = est.ate_interval(X=X[mask], alpha=ALPHA)
    return float(point), float(low), float(high)


# ------------------------------------------------------------------------------ plots
def plot_bias_overview(naive, dml, true_ate):
    labels = ["Naive difference", "LinearDML (DML)", "True effect"]
    vals = [naive["ate"], dml["estimate"], true_ate]
    lows = [naive["ci_low"], dml["ci_low"], true_ate]
    highs = [naive["ci_high"], dml["ci_high"], true_ate]
    err = [np.array(vals) - np.array(lows), np.array(highs) - np.array(vals)]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#c2567a", "#3b6ea8", "#2e8b57"]
    ax.bar(labels, vals, yerr=err, capsize=8, color=colors)
    ax.axhline(true_ate, ls="--", color="#2e8b57", lw=1)
    ax.set_ylabel("Effect on 90-day spend ($)")
    ax.set_title("Loyalty tier upgrade: selection bias corrected by DML")
    fig.tight_layout()
    fig.savefig(FIG / "01_selection_bias_overview.png", dpi=130)
    plt.close(fig)


def plot_overlap(p, T):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(p[T == 1], bins=50, alpha=0.6, density=True, label="Upgrade offered")
    ax.hist(p[T == 0], bins=50, alpha=0.6, density=True, label="Not offered")
    ax.axvline(0.05, ls="--", color="grey")
    ax.axvline(0.95, ls="--", color="grey")
    ax.set_xlabel("Cross-fitted propensity of receiving the upgrade")
    ax.set_title("Overlap check: propensity distributions by group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "02_propensity_overlap.png", dpi=130)
    plt.close(fig)


def plot_heterogeneity_coefficients(coef_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(coef_df))
    ax.errorbar(coef_df["coef"], y, xerr=[coef_df["coef"] - coef_df["ci_low"],
                                          coef_df["ci_high"] - coef_df["coef"]],
                fmt="o", color="#3b6ea8", capsize=4)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, labels=coef_df["term"])
    ax.set_xlabel("Change in treatment effect ($ per unit of term)")
    ax.set_title("LinearDML heterogeneity: what drives the upgrade's value (95% CI)")
    fig.tight_layout()
    fig.savefig(FIG / "03_heterogeneity_coefficients.png", dpi=130)
    plt.close(fig)


def plot_segment_cate(seg_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(seg_df))
    ax.bar(x - 0.2, seg_df["true_ate"], width=0.4, color="#2e8b57", label="True")
    ax.bar(x + 0.2, seg_df["dml_ate"], width=0.4, color="#3b6ea8", label="LinearDML",
           yerr=[seg_df["dml_ate"] - seg_df["dml_ci_low"], seg_df["dml_ci_high"] - seg_df["dml_ate"]],
           capsize=4)
    ax.set_xticks(x, labels=seg_df["property_type"])
    ax.set_ylabel("Effect on 90-day spend ($)")
    ax.set_title("Upgrade effect by property type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "04_cate_by_property_type.png", dpi=130)
    plt.close(fig)


def plot_calibration(pred, true):
    q = pd.qcut(pred, 5, labels=False)
    tab = pd.DataFrame({"pred": pred, "true": true, "q": q}).groupby("q").mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(tab["pred"], tab["true"], "o-", color="#3b6ea8")
    lim = [min(tab.min()), max(tab.max())]
    ax.plot(lim, lim, "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Predicted CATE quintile mean ($)")
    ax.set_ylabel("True CATE quintile mean ($)")
    ax.set_title("CATE calibration (LinearDML, hold-out)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "05_cate_calibration.png", dpi=130)
    plt.close(fig)


def plot_cate_distribution(dml_cate, cf_cate, true_tau):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(true_tau, bins=60, alpha=0.4, density=True, label="True CATE", color="#2e8b57")
    ax.hist(dml_cate, bins=60, alpha=0.4, density=True, label="LinearDML", color="#3b6ea8")
    ax.hist(cf_cate, bins=60, alpha=0.4, density=True, label="Causal Forest", color="#c2567a")
    ax.set_xlabel("Individual effect on 90-day spend ($)")
    ax.set_title("Distribution of individual upgrade effects (hold-out guests)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "06_cate_distribution.png", dpi=130)
    plt.close(fig)


def plot_cf_importance(names, importances):
    imp = pd.Series(importances, index=names).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp.index, imp.values, color="#c2567a")
    ax.set_title("Causal Forest feature importance for effect heterogeneity")
    fig.tight_layout()
    fig.savefig(FIG / "07_causal_forest_importance.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(SEED)
    device = pick_device()
    print(f"XGBoost device: {device}")

    df = simulate(N_GUESTS, rng)
    X, W = build_matrices(df)
    T = df["upgrade_offer"].to_numpy()
    Y = df["spend_90d"].to_numpy()
    true_tau = df["tau_true"].to_numpy()
    true_ate = float(true_tau.mean())

    # ---- summary of the raw data
    print(f"Guests: {len(df):,} | offered upgrade: {T.mean():.1%} | true ATE: ${true_ate:.2f}")
    naive = naive_difference(df)
    print(f"Naive difference: ${naive['ate']:.2f} [{naive['ci_low']:.2f}, {naive['ci_high']:.2f}]")

    # ---- overlap
    p = overlap_propensity(X, W, T, device)
    share_in_support = float(np.mean((p > 0.05) & (p < 0.95)))
    plot_overlap(p, T)

    # ---- hold-out split: fit on train, validate on test
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3, stratify=T, random_state=SEED)
    X_tr, W_tr, T_tr, Y_tr = X.iloc[idx_tr], W.iloc[idx_tr], T[idx_tr], Y[idx_tr]
    X_te, W_te = X.iloc[idx_te].reset_index(drop=True), W.iloc[idx_te].reset_index(drop=True)
    T_te = T[idx_te]
    tau_te = true_tau[idx_te]

    # ---- LinearDML (main estimator)
    print("Fitting LinearDML...")
    dml = fit_linear_dml(X_tr, W_tr, T_tr, Y_tr, device)
    ate_pt = float(dml.ate(X=X_te))
    ate_lo, ate_hi = (float(v) for v in dml.ate_interval(X=X_te, alpha=ALPHA))
    dml_ate = ci_dict(ate_pt, ate_lo, ate_hi)
    dml_cate_te = np.ravel(dml.effect(X_te))

    # heterogeneity table: coefficients on effect modifiers with CIs (feature importance)
    int_lo, int_hi = dml.intercept__interval(alpha=ALPHA)
    coef_lo, coef_hi = dml.coef__interval(alpha=ALPHA)
    names = ["intercept"] + list(X.columns)
    coef_df = pd.DataFrame({
        "term": names,
        "coef": np.r_[np.ravel(dml.intercept_), np.ravel(dml.coef_)],
        "ci_low": np.r_[np.ravel(int_lo), np.ravel(coef_lo)],
        "ci_high": np.r_[np.ravel(int_hi), np.ravel(coef_hi)],
    })
    coef_df["significant"] = ~((coef_df["ci_low"] <= 0) & (coef_df["ci_high"] >= 0))
    coef_df.to_csv(RES / "heterogeneity_coefficients.csv", index=False)
    plot_heterogeneity_coefficients(coef_df.iloc[1:])

    # ---- CausalForestDML (robustness check)
    print("Fitting CausalForestDML...")
    cf = fit_causal_forest(X_tr, W_tr, T_tr, Y_tr, device)
    cf_ate_lo, cf_ate_hi = (float(v) for v in np.ravel(cf.ate_interval(X=X_te, alpha=ALPHA)))
    cf_ate = ci_dict(float(cf.ate(X=X_te)), cf_ate_lo, cf_ate_hi)
    cf_cate_te = np.ravel(cf.effect(X_te))
    plot_cf_importance(list(X.columns), cf.feature_importances_)

    # ---- accuracy against the known truth on hold-out guests
    def accuracy(pred):
        return {
            "rmse": float(np.sqrt(np.mean((pred - tau_te) ** 2))),
            "corr_with_true": float(np.corrcoef(pred, tau_te)[0, 1]),
        }

    dml_acc = accuracy(dml_cate_te)
    cf_acc = accuracy(cf_cate_te)
    plot_calibration(dml_cate_te, tau_te)
    plot_cate_distribution(dml_cate_te, cf_cate_te, tau_te)

    # ---- segment-level effects (property type) with CIs from ATE over each subgroup
    prop_te = df["property_type"].to_numpy()[idx_te]
    seg_rows = []
    for ptype in PROPERTY_TYPES:
        mask = prop_te == ptype
        point, lo_s, hi_s = segment_ate(dml, X_te, mask)
        seg_rows.append({
            "property_type": ptype,
            "n_guests": int(mask.sum()),
            "dml_ate": point,
            "dml_ci_low": lo_s,
            "dml_ci_high": hi_s,
            "true_ate": float(tau_te[mask].mean()),
        })
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(RES / "cate_by_property_type.csv", index=False)
    plot_segment_cate(seg_df)

    # ---- refutations
    print("Running refutations...")
    placebo_T = rng.permutation(T_tr)
    placebo = fit_linear_dml(X_tr, W_tr, placebo_T, Y_tr, device)
    placebo_ate = float(placebo.ate(X=X_te))
    placebo_lo, placebo_hi = (float(v) for v in placebo.ate_interval(X=X_te, alpha=ALPHA))

    W_noise = W_tr.assign(random_common_cause=rng.normal(size=len(W_tr)))
    noisy = fit_linear_dml(X_tr, W_noise, T_tr, Y_tr, device)
    noisy_ate = float(noisy.ate(X=X_te))

    # ---- write results
    validation = {
        "device": device,
        "data": {
            "guests": int(len(df)),
            "train": int(len(idx_tr)),
            "test": int(len(idx_te)),
            "share_offered": float(T.mean()),
            "share_with_overlap_0.05_0.95": share_in_support,
        },
        "true_ate_dollars": true_ate,
        "naive_difference": naive,
        "linear_dml_ate_test": dml_ate,
        "causal_forest_ate_test": cf_ate,
        "linear_dml_cate_accuracy": dml_acc,
        "causal_forest_cate_accuracy": cf_acc,
        "dml_ci_covers_true_ate_test": bool(
            dml_ate["ci_low"] <= float(tau_te.mean()) <= dml_ate["ci_high"]),
        "refutations": {
            "placebo_treatment_ate": placebo_ate,
            "placebo_ci": [placebo_lo, placebo_hi],
            "placebo_ci_covers_zero": bool(placebo_lo <= 0 <= placebo_hi),
            "random_common_cause_ate": noisy_ate,
            "random_common_cause_change": noisy_ate - dml_ate["estimate"],
        },
    }
    (RES / "loyalty_tier_validation.json").write_text(json.dumps(validation, indent=2, default=float))

    pd.DataFrame({
        "guest_id": df["guest_id"].to_numpy()[idx_te],
        "true_cate": tau_te,
        "linear_dml_cate": dml_cate_te,
        "causal_forest_cate": cf_cate_te,
        "propensity": p[idx_te],
    }).to_csv(RES / "test_cate_predictions.csv", index=False)

    # plot 1 uses the test-set DML estimate
    plot_bias_overview(naive, dml_ate, true_ate)

    # ---- console summary
    print("\n=== ATE on 90-day spend ($) ===")
    print(f"True ATE (simulation):          {true_ate:8.2f}")
    print(f"Naive difference:               {naive['ate']:8.2f}  "
          f"[{naive['ci_low']:.2f}, {naive['ci_high']:.2f}]")
    print(f"LinearDML (hold-out):           {dml_ate['estimate']:8.2f}  "
          f"[{dml_ate['ci_low']:.2f}, {dml_ate['ci_high']:.2f}]  covers zero: {dml_ate['covers_zero']}")
    print(f"CausalForestDML (hold-out):     {cf_ate['estimate']:8.2f}  "
          f"[{cf_ate['ci_low']:.2f}, {cf_ate['ci_high']:.2f}]")
    print("\n=== LinearDML heterogeneity (effect modifiers) ===")
    print(coef_df.round(3).to_string(index=False))
    print("\n=== CATE by property type (LinearDML vs truth) ===")
    print(seg_df.round(2).to_string(index=False))
    print("\n=== CATE accuracy (hold-out) ===")
    print(f"LinearDML:       RMSE {dml_acc['rmse']:.2f}, corr {dml_acc['corr_with_true']:.3f}")
    print(f"CausalForestDML: RMSE {cf_acc['rmse']:.2f}, corr {cf_acc['corr_with_true']:.3f}")
    print("\n=== Refutations ===")
    print(f"Placebo treatment ATE: {placebo_ate:.2f} (CI covers zero: "
          f"{placebo_lo <= 0 <= placebo_hi})")
    print(f"Random common cause:   {noisy_ate:.2f} (change {noisy_ate - dml_ate['estimate']:+.2f})")
    print(f"\nOverlap: {share_in_support:.1%} of guests within propensity 0.05-0.95")
    print(f"Figures: {FIG}\nResults: {RES}")


if __name__ == "__main__":
    main()
