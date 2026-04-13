import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.utils import shuffle
import nltk
from nltk.corpus import stopwords
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Consistent aesthetic
PALETTE = {
    "fake":    "#E63946",
    "real":    "#457B9D",
    "accent":  "#F4A261",
    "bg":      "#1D1D2C",
    "surface": "#2A2A3D",
    "text":    "#F1FAEE",
    "muted":   "#8D99AE",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    PALETTE["muted"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["muted"],
    "ytick.color":       PALETTE["muted"],
    "text.color":        PALETTE["text"],
    "grid.color":        "#3A3A55",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    14,
    "axes.labelsize":    11,
    "legend.framealpha": 0.3,
    "legend.facecolor":  PALETTE["surface"],
    "legend.edgecolor":  PALETTE["muted"],
})

def section(title, width=60):
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'═'*width}")

def subsection(title):
    print(f"\n  ┌─ {title}")

def bullet(label, value):
    print(f"  │  {label:<30} {value}")

def save_fig(name, tight=True):
    path = os.path.join(OUTPUT_DIR, name)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.show()
    print(f"  └─ 💾 Saved → {path}")

# ─────────────────────────────────────────────
# NLTK
# ─────────────────────────────────────────────
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
section("1. LOADING DATASET")
df = pd.read_csv("data/15_fake_news_detection.csv")

subsection("Structure")
bullet("Shape", str(df.shape))
bullet("Columns", str(df.columns.tolist()))
print()
print(df.head(3).to_string(index=False))

subsection("Data Types & Missing Values")
for col in df.columns:
    missing = df[col].isnull().sum()
    tag = "⚠️  MISSING" if missing > 0 else "✅"
    print(f"  │  {col:<20} {str(df[col].dtype):<15} {missing} nulls  {tag}")

# ─────────────────────────────────────────────
# 2. LABEL DISTRIBUTION
# ─────────────────────────────────────────────
section("2. LABEL DISTRIBUTION")
counts = df["label"].value_counts()
total  = len(df)
for label, cnt in counts.items():
    bar = "█" * int(cnt / total * 40)
    print(f"  {label.upper():<6} {bar} {cnt} ({cnt/total*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
section("3. FEATURE ENGINEERING")

df["label_num"]   = (df["label"] == "fake").astype(int)
df["text_length"] = df["text"].apply(lambda x: len(str(x)))
df["word_count"]  = df["text"].apply(lambda x: len(str(x).split()))
df["title_length"]= df["title"].apply(lambda x: len(str(x)))

subsection("Descriptive Stats")
print(df[["text_length", "word_count", "title_length"]].describe().round(2).to_string())

# Text cleaning
def clean_and_tokenize(text):
    raw = str(text)
    t   = raw.lower()
    t   = re.sub(r"http\S+|www\S+|https\S+", "", t)
    t   = re.sub(r"<.*?>", "", t)
    t   = re.sub(r"[^a-zA-Z\s]", "", t)
    t   = t.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in t.split() if w not in stop_words]
    return raw, tokens

df[["raw_text", "tokens"]] = df["text"].apply(
    lambda x: pd.Series(clean_and_tokenize(x))
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf    = vectorizer.fit_transform(df["text"])
tfidf_df   = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
df         = pd.concat([df, tfidf_df], axis=1)

# Normalise numeric features
scaler = MinMaxScaler()
df[["text_length_scaled", "word_count_scaled", "title_length_scaled"]] = scaler.fit_transform(
    df[["text_length", "word_count", "title_length"]]
)

subsection("Feature engineering complete")
bullet("Total columns after TF-IDF", str(len(df.columns)))

# ─────────────────────────────────────────────
# 4. VISUALISATIONS  (EDA)
# ─────────────────────────────────────────────
section("4. EXPLORATORY VISUALISATIONS")

# ── 4a. Label Donut + Text-Length Distribution (combined panel) ──────────────
fig = plt.figure(figsize=(14, 5))
fig.suptitle("Fake News Dataset — Exploratory Overview", fontsize=16,
             color=PALETTE["text"], fontweight="bold", y=1.01)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# Donut
ax0 = fig.add_subplot(gs[0])
colors_pie = [PALETTE["fake"], PALETTE["real"]]
wedges, texts, autotexts = ax0.pie(
    counts, labels=counts.index.str.upper(),
    colors=colors_pie, autopct="%1.1f%%",
    pctdistance=0.75, startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=PALETTE["bg"], linewidth=2)
)
for at in autotexts:
    at.set_color(PALETTE["text"]); at.set_fontsize(11)
for t in texts:
    t.set_color(PALETTE["text"]); t.set_fontsize(11)
ax0.set_title("Label Split", color=PALETTE["text"])

# Histogram text length
ax1 = fig.add_subplot(gs[1:])
for label, color in zip(["fake", "real"], [PALETTE["fake"], PALETTE["real"]]):
    subset = df[df["label"] == label]["text_length"]
    ax1.hist(subset, bins=60, alpha=0.65, color=color, label=label.upper(),
             density=True, edgecolor="none")
ax1.set_xlabel("Text Length (characters)")
ax1.set_ylabel("Density")
ax1.set_title("Text Length Distribution by Label")
ax1.legend()
ax1.grid(True)

save_fig("01_eda_overview.png")

# ── 4b. Boxplot text length ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
data_box = [df[df["label"] == l]["text_length"].values for l in ["fake", "real"]]
bp = ax.boxplot(data_box, patch_artist=True,
                medianprops=dict(color=PALETTE["accent"], linewidth=2.5),
                whiskerprops=dict(color=PALETTE["muted"]),
                capprops=dict(color=PALETTE["muted"]),
                flierprops=dict(marker=".", color=PALETTE["muted"], alpha=0.3))
for patch, color in zip(bp["boxes"], [PALETTE["fake"], PALETTE["real"]]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_xticks([1, 2]); ax.set_xticklabels(["FAKE", "REAL"])
ax.set_ylabel("Text Length (characters)")
ax.set_title("Text Length Outlier View (Boxplot)")
ax.grid(True, axis="y")

# Annotate medians
for i, d in enumerate(data_box, 1):
    med = np.median(d)
    ax.text(i + 0.08, med, f"  med={int(med)}", va="center",
            color=PALETTE["accent"], fontsize=9)

save_fig("02_boxplot_text_length.png")

# ── 4c. Correlation heatmap ───────────────────────────────────────────────────
corr = df[["text_length", "word_count", "title_length", "label_num"]].corr()

fig, ax = plt.subplots(figsize=(6, 5))
mask = np.zeros_like(corr, dtype=bool)
np.fill_diagonal(mask, True)   # hide diagonal for cleaner look
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, linecolor=PALETTE["bg"],
            annot_kws={"size": 11}, ax=ax, mask=mask,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix")

save_fig("03_correlation_heatmap.png")

# ─────────────────────────────────────────────
# 5. MODEL TRAINING
# ─────────────────────────────────────────────
section("5. LOGISTIC REGRESSION — TRAINING")

X = df.select_dtypes(include=[np.number]).drop(columns=["label_num"])
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log_reg = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1)
log_reg.fit(X_train, y_train)

subsection("Dataset splits")
bullet("Train samples", str(len(X_train)))
bullet("Test  samples", str(len(X_test)))

# ── Cross-Validation ──────────────────────────────────────────────────────────
section("6. CROSS-VALIDATION (5-FOLD)")
cv_acc = cross_val_score(log_reg, X, y, cv=5, scoring="accuracy",  n_jobs=-1)
cv_auc = cross_val_score(log_reg, X, y, cv=5, scoring="roc_auc",   n_jobs=-1)

subsection("Accuracy per fold")
for i, s in enumerate(cv_acc, 1):
    bar = "▓" * int(s * 30)
    print(f"  │  Fold {i}: {bar} {s:.4f}")
bullet("Mean Accuracy", f"{cv_acc.mean():.4f}  ±{cv_acc.std():.4f}")

subsection("ROC-AUC per fold")
for i, s in enumerate(cv_auc, 1):
    bar = "▓" * int(s * 30)
    print(f"  │  Fold {i}: {bar} {s:.4f}")
bullet("Mean AUC", f"{cv_auc.mean():.4f}  ±{cv_auc.std():.4f}")

# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────
section("7. MODEL EVALUATION")

y_pred  = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)

metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "ROC-AUC": auc}
for name, val in metrics.items():
    stars = "★" * int(val * 5)
    print(f"  {name:<14} {val:.4f}  {stars}")

print(f"\n{classification_report(y_test, y_pred, target_names=['Real','Fake'])}")

# ── 7a. Metrics bar chart ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
colors_bar = [PALETTE["real"], PALETTE["accent"], PALETTE["fake"], "#A8DADC"]
bars = ax.barh(list(metrics.keys()), list(metrics.values()),
               color=colors_bar, edgecolor="none", height=0.55)
ax.set_xlim(0, 1.12)
ax.set_xlabel("Score")
ax.set_title("Model Performance Metrics")
ax.axvline(0.5, color=PALETTE["muted"], linestyle="--", alpha=0.5)
for bar, val in zip(bars, metrics.values()):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", color=PALETTE["text"], fontsize=11)
ax.grid(True, axis="x")
save_fig("04_model_metrics.png")

# ── 7b. Confusion Matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", ax=ax,
            xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
            linewidths=1, linecolor=PALETTE["bg"],
            annot_kws={"size": 16, "weight": "bold"})
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
save_fig("05_confusion_matrix.png")

# ── 7c. ROC Curve ─────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color=PALETTE["accent"], lw=2.5,
        label=f"ROC Curve  (AUC = {auc:.4f})")
ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["accent"])
ax.plot([0, 1], [0, 1], "--", color=PALETTE["muted"], label="Random Classifier")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic (ROC) Curve")
ax.legend(); ax.grid(True)
save_fig("06_roc_curve.png")

# ── 7d. CV scores visualisation ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("5-Fold Cross-Validation Results", fontsize=14,
             color=PALETTE["text"], fontweight="bold")

for ax, scores, title, color in zip(
    axes,
    [cv_acc, cv_auc],
    ["Accuracy", "ROC-AUC"],
    [PALETTE["real"], PALETTE["accent"]]
):
    folds = [f"Fold {i}" for i in range(1, 6)]
    ax.bar(folds, scores, color=color, alpha=0.8, edgecolor="none")
    ax.axhline(scores.mean(), color="white", linestyle="--", linewidth=1.5,
               label=f"Mean = {scores.mean():.4f}")
    ax.set_ylim(max(0, scores.min() - 0.05), min(1.05, scores.max() + 0.05))
    ax.set_title(title); ax.legend(); ax.grid(True, axis="y")
    for i, s in enumerate(scores):
        ax.text(i, s + 0.002, f"{s:.3f}", ha="center",
                color=PALETTE["text"], fontsize=9)

save_fig("07_cv_scores.png")

# ─────────────────────────────────────────────
# 8. SANITY CHECK — SHUFFLED LABELS
# ─────────────────────────────────────────────
section("8. SANITY CHECK — SHUFFLED LABELS")
print("  (If model learns from noise → data leakage detected)\n")

y_shuf = shuffle(y, random_state=42).reset_index(drop=True)
X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
    X, y_shuf, test_size=0.2, random_state=42
)
lr_shuf = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1)
lr_shuf.fit(X_tr_s, y_tr_s)
y_pred_s = lr_shuf.predict(X_te_s)

shuf_metrics = {
    "Accuracy":  accuracy_score(y_te_s, y_pred_s),
    "Precision": precision_score(y_te_s, y_pred_s, zero_division=0),
    "Recall":    recall_score(y_te_s, y_pred_s, zero_division=0),
    "AUC":       roc_auc_score(y_te_s, y_pred_s),
}
print(f"  {'Metric':<14} {'Original':>10}  {'Shuffled':>10}  {'Δ':>10}")
print(f"  {'─'*46}")
for (name, shuf_val), (_, orig_val) in zip(shuf_metrics.items(), metrics.items()):
    delta = orig_val - shuf_val
    flag  = "✅ OK" if abs(shuf_val - 0.5) < 0.1 else "⚠️  CHECK"
    print(f"  {name:<14} {orig_val:>10.4f}  {shuf_val:>10.4f}  {delta:>+10.4f}  {flag}")

# ── Comparison bar chart ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
x  = np.arange(len(shuf_metrics))
w  = 0.35
b1 = ax.bar(x - w/2, list(metrics.values()),     w, label="Original",        color=PALETTE["real"],   alpha=0.85)
b2 = ax.bar(x + w/2, list(shuf_metrics.values()), w, label="Shuffled Labels", color=PALETTE["fake"],   alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(list(metrics.keys()))
ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
ax.set_title("Original vs Shuffled-Label Performance (Data Leakage Sanity Check)")
ax.axhline(0.5, color=PALETTE["muted"], linestyle="--", alpha=0.6, label="Random baseline")
ax.legend(); ax.grid(True, axis="y")

for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center",
                color=PALETTE["text"], fontsize=8)

save_fig("08_sanity_check.png")

# ── Confusion matrix (shuffled) ───────────────────────────────────────────────
cm_s = confusion_matrix(y_te_s, y_pred_s)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_s, annot=True, fmt="g", cmap="Reds", ax=ax,
            xticklabels=["Real","Fake"], yticklabels=["Real","Fake"],
            linewidths=1, linecolor=PALETTE["bg"],
            annot_kws={"size": 16, "weight": "bold"})
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Shuffled Labels (Sanity Check)")
save_fig("09_confusion_matrix_shuffled.png")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
section("✅  ALL DONE")
saved = sorted(os.listdir(OUTPUT_DIR))
print(f"\n  Saved {len(saved)} output file(s) to ./{OUTPUT_DIR}/\n")
for f in saved:
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
    print(f"    📄 {f:<45} {size} KB")
print()
