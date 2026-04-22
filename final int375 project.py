# ============================================================
#  INT375 - CA2 PROJECT | Netflix Dataset
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD ─────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\Asus\Downloads\netflix_titles.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# ── 2. DATA CLEANING ─────────────────────────────────────────
print("\nMissing values BEFORE:")
print(df.isnull().sum())

df["director"]   = df["director"].fillna("Unknown")
df["cast"]       = df["cast"].fillna("Unknown")
df["country"]    = df["country"].fillna(df["country"].mode()[0])
df["rating"]     = df["rating"].fillna(df["rating"].mode()[0])
df["duration"]   = df["duration"].fillna(df["duration"].mode()[0])
df["date_added"] = df["date_added"].fillna(df["date_added"].mode()[0])

df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), errors="coerce")
df["year_added"] = df["date_added"].dt.year

print("\nMissing values AFTER:")
print(df.isnull().sum())

# ── 3. EDA ───────────────────────────────────────────────────
print("\nContent Type:\n",  df["type"].value_counts())
print("\nTop 5 Countries:\n", df["country"].str.split(", ").explode().value_counts().head())
print("\nTitles per Year:\n", df.groupby("year_added")["title"].count().sort_values(ascending=False).head())

movies = df[df["type"] == "Movie"].copy()
movies["minutes"] = movies["duration"].str.extract(r"(\d+)").astype(float)
print("\nMovie Duration → Mean:", round(movies["minutes"].mean(), 1),
      " Max:", movies["minutes"].max(), " Min:", movies["minutes"].min())

# ── 4. OUTLIER DETECTION ─────────────────────────────────────
Q1, Q3  = movies["minutes"].quantile(0.25), movies["minutes"].quantile(0.75)
IQR     = Q3 - Q1
lower   = Q1 - 1.5 * IQR
upper   = Q3 + 1.5 * IQR
outliers = movies[(movies["minutes"] < lower) | (movies["minutes"] > upper)]
print(f"\nIQR Bounds → Lower: {lower:.1f}  Upper: {upper:.1f}")
print(f"Total Outliers   : {len(outliers)}")
print("Skewness         :", round(movies["minutes"].skew(), 2))

# ── 5. CHARTS ────────────────────────────────────────────────

# CHART 1 — BAR: Top 10 Countries
plt.figure(figsize=(10, 5))
top10 = df["country"].str.split(", ").explode().value_counts().head(10)
top10.plot(kind="bar", color="steelblue", edgecolor="black")
plt.plot(range(10), top10.values, "o--", color="red", label="Trend")
plt.title("Top 10 Countries by Title Count")
plt.xlabel("Country")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("chart1_bar.png")
plt.show()

# CHART 2 — PIE: Movies vs TV Shows
plt.figure(figsize=(6, 6))
df["type"].value_counts().plot(kind="pie", autopct="%1.1f%%",
                                colors=["#e50914", "#f5a623"], startangle=90)
plt.title("Movies vs TV Shows")
plt.ylabel("")
plt.tight_layout()
plt.savefig("chart2_pie.png")
plt.show()

# CHART 3 — SCATTER: Release Year vs Duration
plt.figure(figsize=(9, 5))
plt.scatter(movies["release_year"], movies["minutes"], alpha=0.4, color="purple", s=12)
plt.title("Release Year vs Movie Duration")
plt.xlabel("Release Year")
plt.ylabel("Duration (min)")
plt.tight_layout()
plt.savefig("chart3_scatter.png")
plt.show()

# CHART 4 — HISTOGRAM: Movie Duration
plt.figure(figsize=(9, 5))
plt.hist(movies["minutes"].dropna(), bins=30, color="coral", edgecolor="black")
plt.title("Movie Duration Distribution")
plt.xlabel("Duration (min)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("chart4_histogram.png")
plt.show()

# CHART 5 — LINE: Titles Added Per Year
plt.figure(figsize=(10, 5))
yearly = df.groupby(["year_added", "type"]).size().unstack(fill_value=0)
for col, color in zip(yearly.columns, ["#e50914", "#f5a623"]):
    plt.plot(yearly.index, yearly[col], marker="o", linestyle="--",
             color=color, label=col, linewidth=2)
plt.title("Titles Added Per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("chart5_line.png")
plt.show()

# CHART 6 — HEATMAP: Correlation
plt.figure(figsize=(7, 5))
corr = df.select_dtypes(include=np.number).corr()
print("\nCorrelation Matrix:\n", corr)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("chart6_heatmap.png")
plt.show()

# CHART 7 — BOXPLOT: Outlier Detection
plt.figure(figsize=(6, 5))
plt.boxplot(movies["minutes"].dropna())
plt.title("Boxplot: Movie Duration Outliers")
plt.ylabel("Duration (min)")
plt.tight_layout()
plt.savefig("chart7_boxplot.png")
plt.show()

print("\nAll 7 charts saved!")
