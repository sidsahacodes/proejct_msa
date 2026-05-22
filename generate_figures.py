"""
Run from proejct_msa/ to generate all figures and tables for report.tex.
Outputs: figures/fig_eda.pdf, fig_pca.pdf, fig_clusters.pdf, fig_roc.pdf
         tables/tab_classification.tex
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Load and preprocess ──────────────────────────────────────────────────────
df = pd.read_csv("datasets/communities_crime.csv")
outcome_cols = ["ViolentCrimesPerPop", "HighViolentCrime"]
X = df.drop(columns=outcome_cols)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# ── Figure 1: EDA boxplots ───────────────────────────────────────────────────
vars_labels = {
    "medIncome":       "Median Income",
    "PctKids2Par":     "Two-Parent Families",
    "PctPopUnderPov":  "Poverty Rate",
    "PctBSorMore":     "Bachelor's or More",
    "PctHousOwnOcc":   "Owner-Occupied Housing",
    "TotalPctDiv":     "Divorce Rate",
}
fig, axes = plt.subplots(1, 6, figsize=(12, 3.2))
palette = {"Low": "#5B9BD5", "High": "#ED7D31"}
for ax, (var, label) in zip(axes, vars_labels.items()):
    sns.boxplot(data=df, x="HighViolentCrime", y=var, ax=ax,
                order=["Low", "High"], palette=palette,
                width=0.55, fliersize=1.5, linewidth=0.8)
    ax.set_title(label, fontsize=8, pad=3)
    ax.set_xlabel("")
    ax.set_ylabel("Normalized Value" if ax is axes[0] else "", fontsize=8)
    ax.set_xticklabels(["Low", "High"], fontsize=8)
plt.tight_layout(pad=0.5)
plt.savefig("figures/fig_eda.pdf", bbox_inches="tight")
plt.close()
print("fig_eda.pdf saved")

# ── PCA ──────────────────────────────────────────────────────────────────────
pca = PCA()
pca_scores_arr = pca.fit_transform(X_scaled)
pca_scores_df = pd.DataFrame(
    pca_scores_arr, columns=[f"PC{i+1}" for i in range(pca_scores_arr.shape[1])], index=X.index
)
ev = pca.explained_variance_ratio_
pcs_for_80 = int(np.searchsorted(np.cumsum(ev), 0.80)) + 1

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# Scree
n_show = 15
bars = axes[0].bar(range(1, n_show + 1), ev[:n_show] * 100, color="#5B9BD5", alpha=0.75, zorder=2)
ax2 = axes[0].twinx()
ax2.plot(range(1, n_show + 1), np.cumsum(ev[:n_show]) * 100, "r-o",
         markersize=3.5, linewidth=1.2, label="Cumulative")
ax2.axhline(80, color="gray", linestyle="--", linewidth=0.8)
ax2.set_ylabel("Cumulative Variance (%)", fontsize=8)
ax2.tick_params(labelsize=8)
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].set_title("(a) Scree Plot")
ax2.legend(loc="center right", fontsize=7)

# Score plot
for cls, color, alpha in [("Low", "#5B9BD5", 0.35), ("High", "#ED7D31", 0.35)]:
    mask = df["HighViolentCrime"].values == cls
    axes[1].scatter(pca_scores_arr[mask, 0], pca_scores_arr[mask, 1],
                    c=color, alpha=alpha, s=6, label=cls, rasterized=True)
axes[1].set_xlabel(f"PC1 ({ev[0]*100:.1f}% var.)")
axes[1].set_ylabel(f"PC2 ({ev[1]*100:.1f}% var.)")
axes[1].set_title("(b) PC1 vs.\ PC2 Score Plot")
axes[1].legend(title="Crime Class", fontsize=8, title_fontsize=8, markerscale=2)

plt.tight_layout(pad=0.8)
plt.savefig("figures/fig_pca.pdf", bbox_inches="tight")
plt.close()
print("fig_pca.pdf saved")

# ── K-means clustering ────────────────────────────────────────────────────────
cluster_input = pca_scores_df.iloc[:, :pcs_for_80]
km = KMeans(n_clusters=3, n_init=20, random_state=32950)
labels = km.fit_predict(cluster_input)

crime_rate = {k: df.loc[labels == k, "HighViolentCrime"].eq("High").mean() for k in range(3)}
order = sorted(crime_rate, key=crime_rate.get)   # low → high crime
cluster_colors = {order[0]: "#5B9BD5", order[1]: "#70AD47", order[2]: "#ED7D31"}
cluster_names  = {order[0]: f"Low Crime ({crime_rate[order[0]]*100:.0f}\\% High)",
                  order[1]: f"Intermediate ({crime_rate[order[1]]*100:.0f}\\% High)",
                  order[2]: f"High Crime ({crime_rate[order[2]]*100:.0f}\\% High)"}

fig, ax = plt.subplots(figsize=(5.5, 4))
for k in order:
    mask = labels == k
    ax.scatter(pca_scores_arr[mask, 0], pca_scores_arr[mask, 1],
               c=cluster_colors[k], alpha=0.4, s=7,
               label=cluster_names[k], rasterized=True)
ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var.)")
ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var.)")
ax.set_title("K-Means Clusters ($K=3$) in PC Space")
ax.legend(fontsize=8, markerscale=2)
plt.tight_layout()
plt.savefig("figures/fig_clusters.pdf", bbox_inches="tight")
plt.close()
print("fig_clusters.pdf saved")

# ── Classification & ROC curves ───────────────────────────────────────────────
X_fa = X_scaled.copy()
fa = FactorAnalysis(n_components=7, rotation="varimax", random_state=0)
fa.fit(X_fa)
factor_scores = pd.DataFrame(
    fa.transform(X_fa),
    index=X_fa.index,
    columns=[f"Factor{k+1}" for k in range(7)]
)

y = (df["HighViolentCrime"] == "High").astype(int)
train_idx, test_idx = train_test_split(df.index, test_size=0.30, random_state=42, stratify=y)
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

Xo_tr, Xo_te = X_scaled.loc[train_idx], X_scaled.loc[test_idx]
Xp_tr = pca_scores_df.iloc[:, :pcs_for_80].loc[train_idx]
Xp_te = pca_scores_df.iloc[:, :pcs_for_80].loc[test_idx]
Xf_tr, Xf_te = factor_scores.loc[train_idx], factor_scores.loc[test_idx]

def fit_eval(name, model, X_tr, X_te):
    model.fit(X_tr, y_train)
    prob = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {"name": name, "acc": accuracy_score(y_test, pred),
            "sens": tp/(tp+fn), "spec": tn/(tn+fp),
            "auc": roc_auc_score(y_test, prob), "prob": prob}

C_vals = np.logspace(-4, 4, 50)
ridge_cv = LogisticRegressionCV(Cs=C_vals, cv=5, penalty="l2", solver="lbfgs",
                                 scoring="roc_auc", max_iter=5000, random_state=42)
lasso_cv = LogisticRegressionCV(Cs=C_vals, cv=5, penalty="l1", solver="saga",
                                 scoring="roc_auc", max_iter=5000, random_state=42, n_jobs=-1)
elastic_cv = LogisticRegressionCV(Cs=C_vals, cv=5, penalty="elasticnet", solver="saga",
                                   l1_ratios=[0.25, 0.5, 0.75], scoring="roc_auc",
                                   max_iter=5000, random_state=42, n_jobs=-1)
ridge_cv.fit(Xo_tr, y_train);  lasso_cv.fit(Xo_tr, y_train);  elastic_cv.fit(Xo_tr, y_train)

def reg_eval(name, model, X_te):
    prob = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    nz = int((model.coef_.ravel() != 0).sum())
    return {"name": name, "acc": accuracy_score(y_test, pred),
            "sens": tp/(tp+fn), "spec": tn/(tn+fp),
            "auc": roc_auc_score(y_test, prob), "prob": prob, "nz": nz}

results = [
    fit_eval("Logistic (Original)",  LogisticRegression(max_iter=5000), Xo_tr, Xo_te),
    fit_eval("Logistic (PCA)",        LogisticRegression(max_iter=5000), Xp_tr, Xp_te),
    fit_eval("Logistic (Factors)",    LogisticRegression(max_iter=5000), Xf_tr, Xf_te),
    fit_eval("LDA (PCA)",             LinearDiscriminantAnalysis(),       Xp_tr, Xp_te),
    fit_eval("LDA (Factors)",         LinearDiscriminantAnalysis(),       Xf_tr, Xf_te),
    reg_eval("Ridge",  ridge_cv,  Xo_te),
    reg_eval("Lasso",  lasso_cv,  Xo_te),
    reg_eval("Elastic Net", elastic_cv, Xo_te),
]

# ROC figure
fig, ax = plt.subplots(figsize=(5.5, 4.5))
colors_roc = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"]
styles = ["-","-","-","--","--","-.","-.",":"]
for r, color, ls in zip(results, colors_roc, styles):
    fpr, tpr, _ = roc_curve(y_test, r["prob"])
    ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=1.2,
            label=f"{r['name']} ({r['auc']:.3f})")
ax.plot([0,1],[0,1], "k--", linewidth=0.6)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Classification Models")
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig("figures/fig_roc.pdf", bbox_inches="tight")
plt.close()
print("fig_roc.pdf saved")

# ── Classification table (.tex) ───────────────────────────────────────────────
rows = []
for r in results:
    nz_str = f" ({r['nz']})" if "nz" in r else ""
    rows.append(
        f"  {r['name']}{nz_str} & {r['acc']:.3f} & {r['sens']:.3f} & {r['spec']:.3f} & {r['auc']:.3f} \\\\"
    )
# separating line between unpenalised and penalised
rows.insert(5, "  \\midrule")

with open("tables/tab_classification.tex", "w") as f:
    f.write("\n".join(rows) + "\n")
print("tab_classification.tex saved")
print("\nDone.")
