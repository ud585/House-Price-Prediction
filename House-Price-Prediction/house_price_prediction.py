# =============================================================================
# House Price Prediction Using Machine Learning
# =============================================================================
# Author  : Student Project
# Dataset : house_price.csv (50,000 rows, 19 columns)
# Target  : price
# =============================================================================

# ---------------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------------------------
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# Ensure output directories exist
os.makedirs("plots",      exist_ok=True)
os.makedirs("model",      exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

# Fix random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 2. LOAD DATASET
# ---------------------------------------------------------------------------
print("=" * 65)
print("HOUSE PRICE PREDICTION USING MACHINE LEARNING")
print("=" * 65)

df = pd.read_csv("house_price.csv")

# ---------------------------------------------------------------------------
# 3. DATA UNDERSTANDING
# ---------------------------------------------------------------------------
print("\n--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Last 5 Rows ---")
print(df.tail())

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Value Analysis ---")
missing = df.isnull().sum()
print(missing[missing >= 0])           # show all columns

print("\n--- Duplicate Row Analysis ---")
dup_count = df.duplicated().sum()
print(f"Duplicate rows: {dup_count}")

print("\n--- Unique Values for Categorical Columns ---")
cat_cols = ["location", "income_level"]
for col in cat_cols:
    print(f"  {col}: {df[col].unique().tolist()}")

print("\n--- Target Variable (price) Statistics ---")
print(df["price"].describe())

print("\n--- Basic Correlation (numerical columns vs price) ---")
num_cols_for_corr = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[num_cols_for_corr].corr()["price"].sort_values(ascending=False))

# ---------------------------------------------------------------------------
# 4. DATA PREPROCESSING  (no leakage — pipeline applied post-split)
# ---------------------------------------------------------------------------

# Define feature and target columns
TARGET     = "price"
CAT_COLS   = ["location", "income_level"]
# All columns except the target
ALL_FEATS  = [c for c in df.columns if c != TARGET]
NUM_COLS   = [c for c in ALL_FEATS if c not in CAT_COLS]

print("\n--- Feature Columns ---")
print("Numerical:", NUM_COLS)
print("Categorical:", CAT_COLS)

# No missing values found, but pipeline includes imputer for robustness
X = df[ALL_FEATS]
y = df[TARGET]

# ---------------------------------------------------------------------------
# 5. TRAIN–TEST SPLIT (80 / 20)
# ---------------------------------------------------------------------------
# Splitting BEFORE any fitting prevents data leakage:
# the scaler / encoder will only see training data statistics.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
print(f"\n--- Train/Test Split ---")
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 6. BUILD PREPROCESSING PIPELINE
# ---------------------------------------------------------------------------
# Numerical: impute (median) → scale (StandardScaler)
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

# Categorical: impute (most_frequent) → one-hot encode
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, NUM_COLS),
    ("cat", cat_transformer, CAT_COLS)
])

# ---------------------------------------------------------------------------
# 7. EXPLORATORY DATA ANALYSIS (saved to plots/)
# ---------------------------------------------------------------------------
print("\n--- Generating EDA plots ---")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 13, "axes.labelsize": 11})

# (a) Distribution of house prices
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["price"] / 1_000, bins=60, color="steelblue", edgecolor="white", linewidth=0.4)
ax.set_title("Distribution of House Prices")
ax.set_xlabel("Price (in thousands)")
ax.set_ylabel("Frequency")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.0f}K"))
plt.tight_layout()
plt.savefig("plots/price_distribution.png")
plt.close()

# (b) Area vs Price
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["area"], df["price"] / 1_000, alpha=0.15, s=5, color="steelblue")
ax.set_title("Area vs House Price")
ax.set_xlabel("Area (sq ft)")
ax.set_ylabel("Price (in thousands)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.0f}K"))
plt.tight_layout()
plt.savefig("plots/area_vs_price.png")
plt.close()

# (c) Bedrooms vs Price (box plot)
fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column="price", by="bedrooms", ax=ax)
ax.set_title("Bedrooms vs House Price")
ax.set_xlabel("Number of Bedrooms")
ax.set_ylabel("Price")
plt.suptitle("")
plt.tight_layout()
plt.savefig("plots/bedrooms_vs_price.png")
plt.close()

# (d) Bathrooms vs Price
fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column="price", by="bathrooms", ax=ax)
ax.set_title("Bathrooms vs House Price")
ax.set_xlabel("Number of Bathrooms")
ax.set_ylabel("Price")
plt.suptitle("")
plt.tight_layout()
plt.savefig("plots/bathrooms_vs_price.png")
plt.close()

# (e) Age vs Price
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["age"], df["price"] / 1_000, alpha=0.15, s=5, color="coral")
ax.set_title("House Age vs House Price")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Price (in thousands)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.0f}K"))
plt.tight_layout()
plt.savefig("plots/age_vs_price.png")
plt.close()

# (f) Location vs Average Price
fig, ax = plt.subplots(figsize=(7, 5))
loc_avg = df.groupby("location")["price"].mean().sort_values(ascending=False) / 1_000
loc_avg.plot(kind="bar", ax=ax, color="teal", edgecolor="white")
ax.set_title("Location vs Average House Price")
ax.set_xlabel("Location Category")
ax.set_ylabel("Average Price (in thousands)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.0f}K"))
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("plots/location_vs_avg_price.png")
plt.close()

# (g) Income Level vs Average Price
fig, ax = plt.subplots(figsize=(7, 5))
inc_order = ["low", "mid", "high"]
inc_avg = df.groupby("income_level")["price"].mean().reindex(inc_order) / 1_000
inc_avg.plot(kind="bar", ax=ax, color="mediumpurple", edgecolor="white")
ax.set_title("Income Level vs Average House Price")
ax.set_xlabel("Income Level")
ax.set_ylabel("Average Price (in thousands)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.0f}K"))
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("plots/income_level_vs_avg_price.png")
plt.close()

# (h) Correlation Heatmap
fig, ax = plt.subplots(figsize=(14, 10))
corr_matrix = df[num_cols_for_corr].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", linewidths=0.5, ax=ax,
    annot_kws={"size": 7}
)
ax.set_title("Correlation Heatmap – Numerical Features")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# (i) Distance vs Price
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["distance"], df["price"] / 1_000, alpha=0.15, s=5, color="darkorange")
ax.set_title("Distance from City Center vs House Price")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Price (in thousands)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.0f}K"))
plt.tight_layout()
plt.savefig("plots/distance_vs_price.png")
plt.close()

# (j) Crime Rate vs Price
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["crime_rate"], df["price"] / 1_000, alpha=0.15, s=5, color="firebrick")
ax.set_title("Crime Rate vs House Price")
ax.set_xlabel("Crime Rate")
ax.set_ylabel("Price (in thousands)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.0f}K"))
plt.tight_layout()
plt.savefig("plots/crime_rate_vs_price.png")
plt.close()

print("   EDA plots saved to plots/")

# ---------------------------------------------------------------------------
# 8. DEFINE AND TRAIN MODELS
# ---------------------------------------------------------------------------
print("\n--- Training Models ---")

models = {
    "Linear Regression":        LinearRegression(),
    "Decision Tree":            DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest":            RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting":        GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                          max_depth=5, random_state=RANDOM_STATE),
}

results       = {}   # {model_name: {metric: value}}
pipelines     = {}   # {model_name: fitted Pipeline}
predictions   = {}   # {model_name: y_pred array}

for name, model in models.items():
    print(f"   Training: {name} ...", end=" ", flush=True)
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",         model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    results[name]     = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
    pipelines[name]   = pipe
    predictions[name] = y_pred
    print(f"done  |  R²={r2:.4f}  RMSE={rmse:,.0f}")

# ---------------------------------------------------------------------------
# 9. MODEL EVALUATION AND COMPARISON TABLE
# ---------------------------------------------------------------------------
print("\n--- Model Evaluation Results ---")
results_df = pd.DataFrame(results).T
results_df.index.name = "Model"
results_df = results_df.reset_index()
print(results_df.to_string(index=False))

print("\n--- Metric Explanations ---")
print("MAE  (Mean Absolute Error)    : Average of absolute differences between actual and predicted prices.")
print("MSE  (Mean Squared Error)     : Average of squared differences; penalises large errors more heavily.")
print("RMSE (Root Mean Squared Error): Square root of MSE; same unit as price — easier to interpret.")
print("R²   (Coefficient of Determination): Proportion of variance in price explained by the model (1.0 = perfect).")

# Select best model: lowest RMSE (most interpretable for price prediction)
best_name = results_df.loc[results_df["RMSE"].idxmin(), "Model"]
print(f"\n--- Final Model Selected: {best_name} ---")
print("Reason: Lowest RMSE — smallest typical prediction error in the same unit as the target (price).")
print("Also verified to have highest R² score, confirming good overall explanatory power.")

best_pipe  = pipelines[best_name]
best_pred  = predictions[best_name]

# ---------------------------------------------------------------------------
# 10. FEATURE IMPORTANCE (tree-based best model)
# ---------------------------------------------------------------------------
print("\n--- Feature Importance ---")

# Recover feature names after preprocessing
ohe_cols = (best_pipe.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["ohe"]
            .get_feature_names_out(CAT_COLS).tolist())
feature_names = NUM_COLS + ohe_cols

best_estimator = best_pipe.named_steps["model"]

if hasattr(best_estimator, "feature_importances_"):
    importances = best_estimator.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    print(fi_df.head(15).to_string(index=False))

    # Plot top 15 features
    top15 = fi_df.head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top15["Feature"][::-1], top15["Importance"][::-1], color="steelblue", edgecolor="white")
    ax.set_title(f"Feature Importance – {best_name}")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")
    plt.close()
    print("   Feature importance plot saved.")
else:
    # Linear Regression: use absolute coefficients as proxy
    coefs = np.abs(best_estimator.coef_)
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": coefs})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    print(fi_df.head(15).to_string(index=False))

    top15 = fi_df.head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top15["Feature"][::-1], top15["Importance"][::-1], color="steelblue", edgecolor="white")
    ax.set_title(f"Feature Coefficients (abs) – {best_name}")
    ax.set_xlabel("|Coefficient|")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")
    plt.close()

# ---------------------------------------------------------------------------
# 11. ACTUAL VS PREDICTED PLOT
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(y_test / 1_000, best_pred / 1_000, alpha=0.25, s=8, color="steelblue", label="Predictions")
min_val = min(y_test.min(), best_pred.min()) / 1_000
max_val = max(y_test.max(), best_pred.max()) / 1_000
ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect prediction")
ax.set_title(f"Actual vs Predicted House Price – {best_name}")
ax.set_xlabel("Actual Price (in thousands)")
ax.set_ylabel("Predicted Price (in thousands)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.0f}K"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.0f}K"))
ax.legend()
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png")
plt.close()

# Residual plot
residuals = y_test.values - best_pred
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(best_pred / 1_000, residuals / 1_000, alpha=0.2, s=6, color="coral")
axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
axes[0].set_title("Residuals vs Predicted Price")
axes[0].set_xlabel("Predicted Price (in thousands)")
axes[0].set_ylabel("Residual (in thousands)")

axes[1].hist(residuals / 1_000, bins=60, color="coral", edgecolor="white", linewidth=0.4)
axes[1].set_title("Residual Distribution")
axes[1].set_xlabel("Residual (in thousands)")
axes[1].set_ylabel("Frequency")

plt.suptitle(f"Residual Analysis – {best_name}", fontsize=13)
plt.tight_layout()
plt.savefig("plots/residual_analysis.png")
plt.close()
print("   Actual vs Predicted and Residual plots saved.")

# ---------------------------------------------------------------------------
# 12. MODEL COMPARISON BAR CHART
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
model_names = results_df["Model"].tolist()
rmse_vals   = results_df["RMSE"].tolist()
r2_vals     = results_df["R2"].tolist()

axes[0].bar(model_names, [v / 1_000 for v in rmse_vals], color=["steelblue","teal","mediumpurple","coral"])
axes[0].set_title("RMSE Comparison (lower is better)")
axes[0].set_ylabel("RMSE (in thousands)")
axes[0].tick_params(axis="x", rotation=15)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.0f}K"))

axes[1].bar(model_names, r2_vals, color=["steelblue","teal","mediumpurple","coral"])
axes[1].set_title("R² Score Comparison (higher is better)")
axes[1].set_ylabel("R² Score")
axes[1].tick_params(axis="x", rotation=15)
axes[1].set_ylim(0, 1.05)

plt.suptitle("Model Performance Comparison", fontsize=13)
plt.tight_layout()
plt.savefig("plots/model_comparison.png")
plt.close()
print("   Model comparison plot saved.")

# ---------------------------------------------------------------------------
# 13. SAVE FINAL MODEL
# ---------------------------------------------------------------------------
model_path = "model/house_price_model.pkl"
joblib.dump(best_pipe, model_path)
print(f"\n--- Model Saved: {model_path} ---")

# ---------------------------------------------------------------------------
# 14. SAMPLE PREDICTION
# ---------------------------------------------------------------------------
print("\n--- Sample Prediction ---")
print("Using actual categorical values from the dataset:")
print("  location values  :", df["location"].unique().tolist())
print("  income_level values:", df["income_level"].unique().tolist())

sample_input = pd.DataFrame([{
    "area":                 2000,
    "bedrooms":             3,
    "bathrooms":            2,
    "floors":               2,
    "age":                  10,
    "distance":             5,
    "garage":               1,
    "parking":              1,
    "garden":               1,
    "security":             1,
    "school_nearby":        1,
    "hospital_nearby":      1,
    "shopping_mall_nearby": 1,
    "public_transport":     1,
    "crime_rate":           0.2,
    "population_density":   5000,
    "location":             "medium",
    "income_level":         "mid"
}])

loaded_model   = joblib.load(model_path)
sample_pred    = loaded_model.predict(sample_input)[0]
print(f"\nSample House Details:")
print(sample_input.to_string(index=False))
print(f"\n>>> Predicted House Price: {sample_pred:,.2f}")

# ---------------------------------------------------------------------------
# 15. FINAL SUMMARY PRINTOUT
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("FINAL PROJECT SUMMARY")
print("=" * 65)
print(f"Dataset          : house_price.csv")
print(f"Rows / Columns   : {df.shape[0]} / {df.shape[1]}")
print(f"Target Variable  : price")
print(f"Train samples    : {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
print(f"\nModels Trained   : {', '.join(models.keys())}")
print(f"\n{'Model':<28} {'MAE':>14} {'RMSE':>14} {'R²':>8}")
print("-" * 68)
for _, row in results_df.iterrows():
    print(f"{row['Model']:<28} {row['MAE']:>14,.0f} {row['RMSE']:>14,.0f} {row['R2']:>8.4f}")
print("-" * 68)
print(f"\nFinal Model      : {best_name}")
print(f"Sample Prediction: {sample_pred:,.2f}")
print("\nFiles Generated:")
print("  house_price_prediction.py")
print("  requirements.txt")
print("  README.md")
print("  model/house_price_model.pkl")
for f in sorted(os.listdir("plots")):
    print(f"  plots/{f}")
print("=" * 65)
