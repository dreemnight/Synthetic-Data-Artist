# src/main.py
from __future__ import annotations
import argparse, os, json, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ---------------------------
# Data loading & schema
# ---------------------------
def load_or_generate(csv_path: str, seed: int = 42) -> pd.DataFrame:
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    # fallback: generate demo data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1500, n_features=8, n_informative=5, random_state=seed)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(8)])
    df["cat_a"] = pd.qcut(df["num_0"], q=4, labels=["A","B","C","D"]).astype(str)
    df["cat_b"] = np.where(df["num_1"] > 0, "Yes", "No")
    df["target"] = y
    Path("data").mkdir(exist_ok=True, parents=True)
    df.to_csv("data/real_data.csv", index=False)
    return df

def detect_schema(df: pd.DataFrame, categorical_threshold: int = 20):
    numeric_cols, categorical_cols = [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            continue
        # reclassify low-cardinality non-float numerics as categorical
        if df[col].nunique() <= categorical_threshold and not pd.api.types.is_float_dtype(df[col]):
            categorical_cols.append(col)
            numeric_cols.remove(col)
    return numeric_cols, categorical_cols

# ---------------------------
# Gaussian Copula generator
# ---------------------------
from scipy.stats import norm

def _empirical_cdf(x: np.ndarray):
    ranks = x.argsort().argsort().astype(float) + 1.0
    return ranks / (len(x) + 1.0)

def _empirical_ppf(u: np.ndarray, samples: np.ndarray):
    # nearest-quantile for stability across numpy versions
    return np.quantile(samples, u, method="nearest")

def fit_copula(df_num: pd.DataFrame):
    X = df_num.to_numpy()
    U = np.column_stack([_empirical_cdf(X[:, j]) for j in range(X.shape[1])])
    Z = norm.ppf(U)
    Z = np.where(np.isfinite(Z), Z, 0.0)
    corr = np.corrcoef(Z, rowvar=False)
    # tiny regularization -> Cholesky stable
    eps = 1e-6
    corr = (1 - eps) * corr + eps * np.eye(corr.shape[0])
    chol = np.linalg.cholesky(corr)
    return {"chol": chol, "samples": X}

def sample_copula(model: dict, n: int) -> np.ndarray:
    p = model["chol"].shape[0]
    z = np.random.randn(n, p) @ model["chol"].T
    u = norm.cdf(z)
    Xs = model["samples"]
    cols = [ _empirical_ppf(u[:, j], Xs[:, j]) for j in range(p) ]
    return np.column_stack(cols)

def generate_copula(df: pd.DataFrame, numeric_cols, categorical_cols, n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(index=range(n_rows))
    if numeric_cols:
        model = fit_copula(df[numeric_cols])
        out[numeric_cols] = sample_copula(model, n_rows)
    for c in categorical_cols:
        probs = df[c].value_counts(normalize=True)
        out[c] = rng.choice(probs.index.to_numpy(), size=n_rows, p=probs.to_numpy())
    # preserve original column order
    return out[df.columns]

# ---------------------------
# VAE generator (PyTorch)
# ---------------------------
def train_and_generate_vae(
    df: pd.DataFrame, numeric_cols, categorical_cols, n_rows: int,
    seed: int = 42, epochs: int = 30, batch: int = 128, latent: int = 8
) -> pd.DataFrame:
    import torch
    from torch import nn
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    torch.manual_seed(seed); np.random.seed(seed)

    # version-safe OneHotEncoder
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")  # sklearn >= 1.2
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")         # sklearn < 1.2
    scal = StandardScaler()

    X_num = scal.fit_transform(df[numeric_cols]) if numeric_cols else np.empty((len(df), 0))
    X_cat = enc.fit_transform(df[categorical_cols]) if categorical_cols else np.empty((len(df), 0))
    X = np.concatenate([X_num, X_cat], axis=1).astype(np.float32)
    if X.shape[1] == 0:
        raise ValueError("No features to train VAE on (empty schema).")

    class VAE(nn.Module):
        def __init__(self, d, latent=8, hidden=64):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU()
            )
            self.mu = nn.Linear(hidden, latent)
            self.logvar = nn.Linear(hidden, latent)
            self.dec = nn.Sequential(
                nn.Linear(latent, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, d)
            )
        def forward(self, x):
            h = self.enc(x); mu = self.mu(h); logvar = self.logvar(h)
            std = (0.5 * logvar).exp(); eps = torch.randn_like(std)
            z = mu + eps * std
            xr = self.dec(z)
            return xr, mu, logvar

    ds = torch.utils.data.TensorDataset(torch.tensor(X))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    vae = VAE(X.shape[1], latent=latent, hidden=64)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

    def vae_loss(x, xr, mu, logvar):
        recon = nn.MSELoss()(xr, x)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + 1e-3 * kld

    vae.train()
    for _ in range(epochs):
        for (xb,) in dl:
            opt.zero_grad()
            xr, mu, logvar = vae(xb)
            loss = vae_loss(xb, xr, mu, logvar)
            loss.backward(); opt.step()

    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_rows, vae.mu.out_features)
        X_syn = vae.dec(z).cpu().numpy()

    # reconstruct DataFrame (inverse transforms + argmax for categoricals)
    out = pd.DataFrame(index=range(n_rows))
    idx = 0
    if X_num.shape[1] > 0:
        Xn = X_syn[:, idx:idx + X_num.shape[1]]
        Xn = scal.inverse_transform(Xn)
        for i, col in enumerate(numeric_cols):
            out[col] = Xn[:, i]
        idx += X_num.shape[1]
    if X_cat.shape[1] > 0:
        cats = enc.categories_
        start = 0
        for col, cat_vals in zip(categorical_cols, cats):
            k = len(cat_vals)
            block = X_syn[:, idx + start: idx + start + k]
            labels = np.array(cat_vals)[np.argmax(block, axis=1)]
            out[col] = labels
            start += k
        idx += X_cat.shape[1]
    return out[df.columns]

# ---------------------------
# Evaluation & visuals
# ---------------------------
from sklearn.decomposition import PCA
from scipy.spatial.distance import jensenshannon

def _jsd(a, b, bins=30):
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    b = (b - b.min()) / (b.max() - b.min() + 1e-9)
    pa, _ = np.histogram(a, bins=bins, range=(0,1), density=True)
    pb, _ = np.histogram(b, bins=bins, range=(0,1), density=True)
    pa = pa / (pa.sum() + 1e-12); pb = pb / (pb.sum() + 1e-12)
    return float(jensenshannon(pa, pb))

def plot_distribution_overlap(df_real, df_syn, bins, out_path: Path):
    num_cols = [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c])]
    if not num_cols: return {}
    cols = num_cols[:min(6, len(num_cols))]
    fig, axes = plt.subplots(len(cols), 1, figsize=(8, 2.2*len(cols)))
    if len(cols) == 1: axes = [axes]
    scores = {}
    for ax, c in zip(axes, cols):
        ax.hist(df_real[c].dropna(), bins=bins, alpha=0.5, label="real", density=True)
        ax.hist(df_syn[c].dropna(), bins=bins, alpha=0.5, label="synthetic", density=True)
        ax.set_title(f"Distribution overlap: {c}")
        ax.legend()
        scores[c] = 1.0 - _jsd(df_real[c].dropna().to_numpy(), df_syn[c].dropna().to_numpy(), bins=bins)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return scores

def plot_pca(df_real, df_syn, out_path: Path, n_components=2):
    # silence harmless sklearn warning about feature names
    warnings.filterwarnings("ignore", message=".*feature names.*PCA.*")
    num_cols = [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c])]
    if not num_cols: return None
    Xr = df_real[num_cols].fillna(df_real[num_cols].mean())
    Xs = df_syn[num_cols].fillna(df_syn[num_cols].mean())
    X = pd.concat([Xr, Xs], axis=0).to_numpy()
    pca = PCA(n_components=n_components, random_state=42).fit(X)
    Zr = pca.transform(Xr); Zs = pca.transform(Xs)
    plt.figure(figsize=(7,5))
    plt.scatter(Zr[:,0], Zr[:,1], s=10, alpha=0.5, label="real")
    plt.scatter(Zs[:,0], Zs[:,1], s=10, alpha=0.5, label="synthetic")
    plt.title("PCA Projection: real vs synthetic")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return {"explained_variance": pca.explained_variance_ratio_.tolist()}

def plot_correlation_heatmap(df_real, df_syn, out_path: Path):
    num_cols = [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c])]
    if not num_cols: return {}
    corr_r = df_real[num_cols].corr().to_numpy()
    corr_s = df_syn[num_cols].corr().to_numpy()
    diff = float(np.abs(corr_r - corr_s).mean())
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    sns.heatmap(corr_r, ax=axes[0], vmin=-1, vmax=1, cmap="coolwarm", cbar=False); axes[0].set_title("Real correlation")
    sns.heatmap(corr_s, ax=axes[1], vmin=-1, vmax=1, cmap="coolwarm", cbar=False); axes[1].set_title("Synthetic correlation")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return {"correlation_diff_mean": diff}

def pairplot_compare(df_real, df_syn, out_path: Path, sample=500):
    num_cols = [c for c in df_real.columns if pd.api.types.is_numeric_dtype(df_real[c])]
    if len(num_cols) < 2: return False
    cols = num_cols[:min(4, len(num_cols))]
    R = df_real[cols].sample(n=min(sample, len(df_real)), random_state=42).copy(); R["__type__"] = "real"
    S = df_syn[cols].sample(n=min(sample, len(df_syn)), random_state=42).copy(); S["__type__"] = "synthetic"
    Z = pd.concat([R, S], axis=0)
    g = sns.pairplot(Z, vars=cols, hue="__type__", plot_kws={"alpha":0.5, "s":12}, diag_kind="hist")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path); plt.close("all")
    return True

# ---------------------------
# Report writer (per-run)
# ---------------------------
def write_report(method: str, rows: int, seed: int, metrics: dict, report_path: Path, run_name: str):
    base = f"../outputs/{run_name}/plots"
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Synthetic Data Artist - Report</title>
<style>body{{font-family:Arial;margin:24px}} pre{{background:#f6f8fa;padding:12px}}</style></head>
<body>
<h1>Synthetic Data Artist</h1>
<p>Method: <b>{method}</b> · Rows: {rows} · Seed: {seed}</p>
<h2>Metrics</h2><pre>{json.dumps(metrics, indent=2)}</pre>
<h2>Distribution Overlap</h2><img src="{base}/distribution_overlap.png">
<h2>PCA Projection</h2><img src="{base}/pca_projection.png">
<h2>Correlation</h2><img src="{base}/correlation_heatmap.png">
<h2>Pairplot</h2><img src="{base}/pairplot_comparison.png">
</body></html>"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--method", choices=["copula", "vae"], default="copula")
    ap.add_argument("--data", default="data/real_data.csv")
    ap.add_argument("--run_name", default=None, help="Optional name to separate outputs")
    args = ap.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    seed = int(cfg.get("seed", 42)); set_seed(seed)

    rows_cfg = cfg.get("rows", None)
    bins = int(cfg.get("hist_bins", 30))
    pca_components = int(cfg.get("pca_components", 2))
    pairplot_sample = int(cfg.get("pairplot_sample", 500))
    cat_thr = int(cfg.get("categorical_threshold", 20))

    run_name = args.run_name or args.method
    data_dir = Path("data")
    run_dir = Path("outputs") / run_name
    plots_dir = run_dir / "plots"
    reports_dir = Path("reports")

    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_generate(args.data, seed=seed)
    numeric_cols, categorical_cols = detect_schema(df, categorical_threshold=cat_thr)
    n_rows = int(rows_cfg) if rows_cfg is not None else len(df)

    if args.method == "copula":
        df_syn = generate_copula(df, numeric_cols, categorical_cols, n_rows=n_rows, seed=seed)
    else:
        df_syn = train_and_generate_vae(
            df, numeric_cols, categorical_cols, n_rows=n_rows,
            seed=seed, epochs=30, batch=128, latent=8
        )

    syn_path = data_dir / f"synthetic_data_{run_name}.csv"
    df_syn.to_csv(syn_path, index=False)

    # Metrics + plots
    dist_scores = plot_distribution_overlap(df, df_syn, bins=bins, out_path=plots_dir / "distribution_overlap.png")
    pca_info = plot_pca(df, df_syn, out_path=plots_dir / "pca_projection.png", n_components=pca_components)
    corr_info = plot_correlation_heatmap(df, df_syn, out_path=plots_dir / "correlation_heatmap.png")
    _ = pairplot_compare(df, df_syn, out_path=plots_dir / "pairplot_comparison.png", sample=pairplot_sample)

    metrics = {
        "rows_real": int(len(df)),
        "rows_synthetic": int(len(df_syn)),
        "method": args.method,
        "seed": seed,
        "distribution_overlap_mean": float(np.mean(list(dist_scores.values()))) if dist_scores else None,
        "distribution_overlap_per_feature": dist_scores,
        "pca_explained_variance": (pca_info or {}).get("explained_variance"),
        **corr_info
    }
    save_json(run_dir / "metrics.json", metrics)

    write_report(
        method=args.method, rows=n_rows, seed=seed, metrics=metrics,
        report_path=reports_dir / f"{run_name}_report.html", run_name=run_name
    )

    print(f"Saved synthetic CSV: {syn_path}")
    print(f"Saved metrics:       {run_dir / 'metrics.json'}")
    print(f"Saved plots in:      {plots_dir}")
    print(f"Saved report:        {reports_dir / f'{run_name}_report.html'}")

if __name__ == "__main__":
    main()
