# # NOTES TO RUN THIS APP LOCALLY:
# # app.py
# # EPL Match Performance & Home Advantage Dashboard (tabbed version + COVID attendance handling)
# # Data file: mydata.csv (place in the SAME directory as this app.py)
# #
# # Run:
# #   pip install pandas numpy matplotlib scipy statsmodels streamlit
# #   streamlit run app.py

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
# import statsmodels.api as sm

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # -------------------------
# # Config
# # ------------------------
CSV_PATH = "mydata.csv"

st.set_page_config(page_title="EPL Match Performance & Home Advantage", layout="wide")


# -------------------------
# Cleaning helpers (consistent naming)
# -------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Matches teammate: df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def parse_mixed_dates(s: pd.Series) -> pd.Series:
    """
    Robust parsing for both:
      - '28th May 2023'
      - '23/05/2021' (DD/MM/YYYY)
    """
    x = s.astype(str).str.strip()

    # Remove ordinal suffixes: 1st/2nd/3rd/4th... -> 1/2/3/4...
    x = x.str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True)

    # Slash dates
    mask_slash = x.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$", na=False)

    out = pd.Series(pd.NaT, index=x.index, dtype="datetime64[ns]")
    out.loc[mask_slash] = pd.to_datetime(x.loc[mask_slash], format="%d/%m/%Y", errors="coerce")
    out.loc[~mask_slash] = pd.to_datetime(x.loc[~mask_slash], dayfirst=True, errors="coerce")
    return out


def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the same targets/diffs the teammate notebook uses:
      - year/month/day, season
      - home_win_binary, away_win_binary, draw_binary, close_match
      - possession_diff, shots_diff, shots_on_target_diff, fouls_diff, yellow_diff, red_diff, goal_diff
      - attendance_high
      - closed_doors (attendance==0) for COVID period
    """
    df = df.copy()
    df = standardize_columns(df)

    # Attendance cleaning ( remove commas then numeric) 
    if "attendance" in df.columns:
        df["attendance"] = (
            df["attendance"]
            .astype(str)
            .str.replace('"', "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")

    # Date cleaning (robust)
    if "date" in df.columns:
        df["date"] = parse_mixed_dates(df["date"])
        # If any failures, leave as NaT; dashboard will still run but filters may be limited
        df["year"] = df["date"].dt.year.astype("Int64")
        df["month"] = df["date"].dt.month.astype("Int64")
        df["day"] = df["date"].dt.day.astype("Int64")

        # Build EPL season label (season starts ~Aug)
        # Column name "season" 
        season_start = np.where(df["month"] >= 8, df["year"], df["year"] - 1)
        season_start = pd.Series(season_start, index=df.index).astype("Int64")
        df["season"] = season_start.astype(str) + "-" + (season_start + 1).astype(str)

    # Ensure goals numeric
    for c in ["goals_home", "away_goals"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Match outcome (H/D/A) from class 
    if "class" in df.columns:
        df["outcome_hda"] = df["class"].astype(str).str.lower().map({"h": "H", "d": "D", "a": "A"})
    else:
        # fallback from goals
        df["outcome_hda"] = np.where(
            df["goals_home"] > df["away_goals"],
            "H",
            np.where(df["goals_home"] == df["away_goals"], "D", "A"),
        )

    # Binary targets
    df["home_win_binary"] = (df["goals_home"] > df["away_goals"]).astype(int)
    df["away_win_binary"] = (df["away_goals"] > df["goals_home"]).astype(int)
    df["draw_binary"] = (df["goals_home"] == df["away_goals"]).astype(int)
    df["close_match"] = (abs(df["goals_home"] - df["away_goals"]) <= 1).astype(int)

    # Human-readable match result (for charts)
    df["match_result"] = np.where(
        df["home_win_binary"] == 1, "Home Win",
        np.where(df["away_win_binary"] == 1, "Away Win", "Draw")
    )

    # Core diffs (consistent names) 
    if "home_possessions" in df.columns and "away_possessions" in df.columns:
        df["home_possessions"] = pd.to_numeric(df["home_possessions"], errors="coerce")
        df["away_possessions"] = pd.to_numeric(df["away_possessions"], errors="coerce")
        df["possession_diff"] = df["home_possessions"] - df["away_possessions"]

    if "home_shots" in df.columns and "away_shots" in df.columns:
        df["shots_diff"] = df["home_shots"] - df["away_shots"]

    if "home_on" in df.columns and "away_on" in df.columns:
        df["shots_on_target_diff"] = df["home_on"] - df["away_on"]

    if "home_fouls" in df.columns and "away_fouls" in df.columns:
        df["fouls_diff"] = df["home_fouls"] - df["away_fouls"]

    if "home_yellow" in df.columns and "away_yellow" in df.columns:
        df["yellow_diff"] = df["home_yellow"] - df["away_yellow"]

    if "home_red" in df.columns and "away_red" in df.columns:
        df["red_diff"] = df["home_red"] - df["away_red"]

    df["goal_diff"] = df["goals_home"] - df["away_goals"]

    # Attendance helpers
    if "attendance" in df.columns:
        df["attendance_high"] = (df["attendance"] > df["attendance"].median()).astype(int)
        df["closed_doors"] = df["attendance"].fillna(0).eq(0).astype(int)
    else:
        df["attendance_high"] = 0
        df["closed_doors"] = 0

    return df


# -------------------------
# Stats helpers (used in tabs)
# -------------------------
def corr_with_target(df: pd.DataFrame, target="home_win_binary", features=None, method="pearson") -> pd.DataFrame:
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in features if c != target]

    out = []
    for c in features:
        s = df[[c, target]].dropna()
        if s.empty or s[c].nunique() < 2:
            continue
        if method == "spearman":
            r, _ = stats.spearmanr(s[c], s[target])
        else:
            r = np.corrcoef(s[c], s[target])[0, 1]
        out.append((c, r, len(s)))

    res = pd.DataFrame(out, columns=["feature", "corr", "n"]).sort_values(
        "corr", key=lambda x: x.abs(), ascending=False
    )
    return res


def cohens_d(x, y) -> float:
    x, y = np.array(x), np.array(y)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled if pooled > 0 else np.nan


def bootstrap_ci_diff_means(x, y, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    x = np.array(x); y = np.array(y)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return (np.nan, np.nan, np.nan)

    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs[i] = xb.mean() - yb.mean()

    alpha = (1 - ci) / 2
    lo = np.quantile(diffs, alpha)
    hi = np.quantile(diffs, 1 - alpha)
    return (x.mean() - y.mean(), lo, hi)


def compare_groups(df: pd.DataFrame, feature: str, target="home_win_binary", n_boot=2000) -> dict:
    s = df[[feature, target]].dropna()
    g1 = s[s[target] == 1][feature]
    g0 = s[s[target] == 0][feature]

    if g1.shape[0] < 2 or g0.shape[0] < 2:
        return {
            "feature": feature,
            "n_home_win": int(g1.shape[0]),
            "n_not_home_win": int(g0.shape[0]),
            "mean_home_win": float(g1.mean()) if g1.shape[0] else np.nan,
            "mean_not_home_win": float(g0.mean()) if g0.shape[0] else np.nan,
            "mean_diff": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "welch_t_pvalue": np.nan,
            "cohens_d": np.nan,
        }

    _, p = stats.ttest_ind(g1, g0, equal_var=False, nan_policy="omit")
    d = cohens_d(g1, g0)
    mean_diff, lo, hi = bootstrap_ci_diff_means(g1, g0, n_boot=n_boot)

    return {
        "feature": feature,
        "n_home_win": int(g1.shape[0]),
        "n_not_home_win": int(g0.shape[0]),
        "mean_home_win": float(g1.mean()),
        "mean_not_home_win": float(g0.mean()),
        "mean_diff": float(mean_diff),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "welch_t_pvalue": float(p),
        "cohens_d": float(d),
    }


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(pvals.size, np.nan, dtype=float)
    valid = np.isfinite(pvals)
    pv = pvals[valid]
    if pv.size == 0:
        return out

    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * (pv.size / (np.arange(1, pv.size + 1)))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    out_valid = np.empty_like(pv)
    out_valid[order] = adj
    out[valid] = out_valid
    return out


def binned_win_rate(df: pd.DataFrame, xcol: str, bins: int = 10) -> pd.DataFrame:
    s = df[[xcol, "home_win_binary"]].dropna()
    if s.empty or s[xcol].nunique() < bins:
        return pd.DataFrame()
    s = s.copy()
    s["bin"] = pd.qcut(s[xcol], q=bins, duplicates="drop")
    out = s.groupby("bin", observed=True).agg(
        n=("home_win_binary", "size"),
        home_win_rate=("home_win_binary", "mean"),
        x_mean=(xcol, "mean"),
    ).reset_index()
    return out


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    raw = pd.read_csv(CSV_PATH)
    return add_core_features(raw)


# -------------------------
# Load & header
# -------------------------
st.title("EPL Match Performance & Home Advantage (Interactive)")
st.caption("Dashboard uses teammate-aligned columns/features + robust mixed-date parsing.")

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Could not find '{CSV_PATH}'. Put mydata.csv in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# data check
with st.expander("Data consistency and validation checks (optional)"):
    st.write("Rows:", len(df))
    if "date" in df.columns:
        st.write("Unparsed dates (NaT):", int(df["date"].isna().sum()))
    if "attendance" in df.columns:
        st.write("Closed-door matches (attendance==0):", int((df["attendance"] == 0).sum()))
        
        
# -------------------------
# Models: Logistic Regression + Gradient Boosting 
# -------------------------
@st.cache_data(show_spinner=False)
def fit_lr_gb_models(df_in: pd.DataFrame):
    """
    Fit Logistic Regression + Gradient Boosting using the teammate's selected feature set.
    Returns holdout metrics + time-based CV metrics + importance summaries.
    """
    dfm = df_in.copy()

    # selected features 
    features = [
        "possession_diff",
        "shots_diff",
        "shots_on_target_diff",
        "fouls_diff",
        "yellow_diff",
        "red_diff",
        "attendance",
        "attendance_high",
    ]

    # Required columns
    required = ["home_win_binary", "year", "date"] + features
    missing = [c for c in required if c not in dfm.columns]
    if missing:
        return {"error": f"Missing required columns for models tab: {missing}"}

    # Drop NA rows for modeling
    dfm = dfm[required].dropna().copy()

    # Ensure date is datetime for sorting (your app already parses this robustly)
    dfm["date"] = pd.to_datetime(dfm["date"], errors="coerce")
    dfm = dfm.dropna(subset=["date"])

    # Sort for time-based splits
    dfm = dfm.sort_values(["year", "date"]).reset_index(drop=True)

    X = dfm[features].astype(float)
    y = dfm["home_win_binary"].astype(int)

    # -------------------------
    # Holdout split: year < 2023 train, year >= 2023 test 
    # -------------------------
    train_mask = dfm["year"] < 2023
    test_mask = dfm["year"] >= 2023

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    # Logistic Regression (holdout)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_holdout = LogisticRegression(
        max_iter=1000,
        C=10,
        solver="lbfgs",
        class_weight="balanced",
    )
    lr_holdout.fit(X_train_scaled, y_train)
    lr_pred = lr_holdout.predict(X_test_scaled)

    lr_holdout_acc = accuracy_score(y_test, lr_pred)
    lr_holdout_cm = confusion_matrix(y_test, lr_pred)
    lr_holdout_report = classification_report(y_test, lr_pred, output_dict=True)

    lr_holdout_coef = pd.DataFrame({
        "Feature": features,
        "Coefficient": lr_holdout.coef_[0]
    }).assign(AbsCoeff=lambda d: d["Coefficient"].abs()).sort_values("AbsCoeff", ascending=False)

    # Gradient Boosting (holdout) 
    gb_holdout = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    gb_holdout.fit(X_train, y_train)
    gb_pred = gb_holdout.predict(X_test)

    gb_holdout_acc = accuracy_score(y_test, gb_pred)
    gb_holdout_cm = confusion_matrix(y_test, gb_pred)
    gb_holdout_report = classification_report(y_test, gb_pred, output_dict=True)

    gb_holdout_imp = pd.DataFrame({
        "Feature": features,
        "Importance": gb_holdout.feature_importances_
    }).sort_values("Importance", ascending=False)

    # -------------------------
    # Time-based CV (TimeSeriesSplit)
    # -------------------------
    tscv = TimeSeriesSplit(n_splits=4)

    lr_fold_acc = []
    lr_coef_list = []
    lr_all_true, lr_all_pred = [], []

    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        lr = LogisticRegression(max_iter=1000, C=10, solver="lbfgs", class_weight="balanced")
        lr.fit(X_tr_s, y_tr)
        pred = lr.predict(X_te_s)

        lr_fold_acc.append(accuracy_score(y_te, pred))
        lr_coef_list.append(lr.coef_[0])
        lr_all_true.extend(y_te.tolist())
        lr_all_pred.extend(pred.tolist())

    lr_avg_coef = np.mean(np.array(lr_coef_list), axis=0)
    lr_cv_coef = pd.DataFrame({
        "Feature": features,
        "AvgCoefficient": lr_avg_coef
    }).assign(AbsCoeff=lambda d: d["AvgCoefficient"].abs()).sort_values("AbsCoeff", ascending=False)

    lr_cv_cm = confusion_matrix(lr_all_true, lr_all_pred)
    lr_cv_report = classification_report(lr_all_true, lr_all_pred, output_dict=True)

    gb_fold_acc = []
    gb_imp_list = []
    gb_all_true, gb_all_pred = [], []

    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        gb.fit(X_tr, y_tr)
        pred = gb.predict(X_te)

        gb_fold_acc.append(accuracy_score(y_te, pred))
        gb_imp_list.append(gb.feature_importances_)
        gb_all_true.extend(y_te.tolist())
        gb_all_pred.extend(pred.tolist())

    gb_avg_imp = np.mean(np.array(gb_imp_list), axis=0)
    gb_cv_imp = pd.DataFrame({
        "Feature": features,
        "AvgImportance": gb_avg_imp
    }).sort_values("AvgImportance", ascending=False)

    gb_cv_cm = confusion_matrix(gb_all_true, gb_all_pred)
    gb_cv_report = classification_report(gb_all_true, gb_all_pred, output_dict=True)

    return {
        "features": features,
        "n_rows_used": len(dfm),

        "lr_holdout_acc": lr_holdout_acc,
        "lr_holdout_cm": lr_holdout_cm,
        "lr_holdout_report": lr_holdout_report,
        "lr_holdout_coef": lr_holdout_coef,

        "gb_holdout_acc": gb_holdout_acc,
        "gb_holdout_cm": gb_holdout_cm,
        "gb_holdout_report": gb_holdout_report,
        "gb_holdout_imp": gb_holdout_imp,

        "lr_cv_fold_acc": lr_fold_acc,
        "lr_cv_mean_acc": float(np.mean(lr_fold_acc)),
        "lr_cv_cm": lr_cv_cm,
        "lr_cv_report": lr_cv_report,
        "lr_cv_coef": lr_cv_coef,

        "gb_cv_fold_acc": gb_fold_acc,
        "gb_cv_mean_acc": float(np.mean(gb_fold_acc)),
        "gb_cv_cm": gb_cv_cm,
        "gb_cv_report": gb_cv_report,
        "gb_cv_imp": gb_cv_imp,
    }


# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")
dff = df.copy()

filter_mode = st.sidebar.radio("Filter by", ["Year", "Season"], index=0)

if filter_mode == "Year" and "year" in dff.columns:
    years = sorted([int(y) for y in dff["year"].dropna().unique()])
    year_sel = st.sidebar.multiselect("Year", years, default=years)
    if year_sel:
        dff = dff[dff["year"].isin(year_sel)]

if filter_mode == "Season" and "season" in dff.columns:
    seasons = sorted([s for s in dff["season"].dropna().unique()])
    season_sel = st.sidebar.multiselect("Season", seasons, default=seasons)
    if season_sel:
        dff = dff[dff["season"].isin(season_sel)]

# Team filter (teams are numeric IDs in dataset)
if "home_team" in dff.columns and "away_team" in dff.columns:
    teams = sorted(pd.unique(pd.concat([dff["home_team"], dff["away_team"]], ignore_index=True)))
    team_sel = st.sidebar.selectbox("Team (optional)", ["(All)"] + [str(t) for t in teams])
    if team_sel != "(All)":
        dff = dff[(dff["home_team"].astype(str) == team_sel) | (dff["away_team"].astype(str) == team_sel)]

# Outcome filter
outcome_sel = st.sidebar.multiselect("Outcome (H/D/A)", ["H", "D", "A"], default=["H", "D", "A"])
if outcome_sel and "outcome_hda" in dff.columns:
    dff = dff[dff["outcome_hda"].isin(outcome_sel)]

st.sidebar.divider()
st.sidebar.caption("Tip: If charts look sparse, widen filters (all years/seasons/teams).")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Home Advantage", "Correlations & Heatmap", "Attendance & Possession", "Association Tests", "Trends & Binning", "Models (LR + GB)"]
)

# -------------------------
# Tab 1: Home Advantage
# -------------------------
# -------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(dff):,}")
    c2.metric("Home Win Rate", f"{dff['home_win_binary'].mean():.3f}")
    c3.metric("Draw Rate", f"{dff['draw_binary'].mean():.3f}")
    c4.metric("Avg Goal Diff (Home-Away)", f"{dff['goal_diff'].mean():.3f}")

    st.subheader("Goal Difference Distribution (Home - Away)")
    fig = plt.figure(figsize=(10, 4))
    plt.hist(dff["goal_diff"].dropna(), bins=20)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Goal difference (goals_home - away_goals)")
    plt.ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Outcome Rates (Home Win / Draw / Away Win)")
    rates = (
        dff["match_result"]
        .value_counts(normalize=True)
        .reindex(["Home Win", "Draw", "Away Win"])
        .fillna(0)
    )
    fig2 = plt.figure(figsize=(6, 4))
    plt.bar(rates.index, rates.values)
    plt.ylim(0, 1)
    plt.ylabel("Proportion")
    plt.title("Overall Match Outcome Rates")
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("### Match outcomes by year (H/D/A)")

    if {"year", "outcome_hda"}.issubset(dff.columns):
        tmp = dff.dropna(subset=["year", "outcome_hda"]).copy()

        if len(tmp) == 0:
            st.info("No year/outcome data available under current filters.")
        else:
            tmp["year"] = tmp["year"].astype(int)

            counts = (
                tmp.groupby(["year", "outcome_hda"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=["H", "D", "A"], fill_value=0)
                .sort_index()
            )

            mode = st.radio(
                "Display yearly outcomes as",
                ["Counts", "Proportions"],
                horizontal=True,
                key="tab1_outcome_mode"
            )

            years = counts.index.tolist()
            x = np.arange(len(years))
            w = 0.25

            if mode == "Proportions":
                plot_tbl = counts.div(counts.sum(axis=1), axis=0)
                ylabel = "Proportion of matches"
                ylim = (0, 1)
                title = "Match outcomes by year (proportions)"
            else:
                plot_tbl = counts
                ylabel = "Number of matches"
                ylim = None
                title = "Match outcomes by year (counts)"

            fig3 = plt.figure(figsize=(10, 4))
            plt.bar(x - w, plot_tbl["H"].values, width=w, label="H")
            plt.bar(x,      plot_tbl["D"].values, width=w, label="D")
            plt.bar(x + w,  plot_tbl["A"].values, width=w, label="A")
            plt.xticks(x, years)
            plt.ylabel(ylabel)
            plt.title(title)
            if ylim is not None:
                plt.ylim(*ylim)
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig3)

            st.dataframe(counts, use_container_width=True)
    else:
        st.info("Need year and outcome_hda to draw the grouped outcome chart.")

    st.markdown("### Closed doors vs with crowd (COVID context)")

    needed = {"closed_doors", "home_win_binary", "goal_diff"}
    if needed.issubset(dff.columns):
        tmp = dff.dropna(subset=list(needed)).copy()
        tmp["closed_doors"] = tmp["closed_doors"].astype(int)

        if tmp["closed_doors"].nunique() < 2:
            st.info(
                "Current filters include only one attendance condition "
                "(all closed doors or all with crowd). Select more years to compare both."
            )
        else:
            summary = (
                tmp.groupby("closed_doors")
                .agg(
                    n=("home_win_binary", "size"),
                    home_win_rate=("home_win_binary", "mean"),
                    avg_goal_diff=("goal_diff", "mean"),
                )
                .sort_index()
            )

            labels = ["Closed doors\n(attendance = 0)", "With crowd\n(attendance > 0)"]
            home_win_rates = [summary.loc[1, "home_win_rate"], summary.loc[0, "home_win_rate"]]
            goal_diffs = [summary.loc[1, "avg_goal_diff"], summary.loc[0, "avg_goal_diff"]]

            col1, col2 = st.columns(2)

            with col1:
                fig4 = plt.figure(figsize=(6, 4))
                plt.bar(labels, home_win_rates)
                plt.ylim(0, 1)
                plt.ylabel("Home win rate")
                plt.title("Home win rate")
                plt.tight_layout()
                st.pyplot(fig4)

            with col2:
                fig5 = plt.figure(figsize=(6, 4))
                plt.bar(labels, goal_diffs)
                plt.axhline(0, linewidth=1)
                plt.ylabel("Avg goal diff (Home - Away)")
                plt.title("Average goal difference")
                plt.tight_layout()
                st.pyplot(fig5)

            st.dataframe(summary, use_container_width=True)
    else:
        st.info("Need closed_doors, home_win_binary, and goal_diff to draw this comparison.")

    st.caption("These are descriptive summaries (association, not causation).")



# -------------------------
# Tab 2: Correlations + Heatmap
# -------------------------
with tab2:
    st.subheader("Correlations with home_win_binary (association)")

    # Candidate features: diffs + a few base numeric signals
    diff_cols = [c for c in dff.columns if c.endswith("_diff") and pd.api.types.is_numeric_dtype(dff[c])]
    base_cols = [c for c in ["attendance", "home_possessions", "away_possessions", "home_shots", "away_shots", "home_on", "away_on"]
                 if c in dff.columns and pd.api.types.is_numeric_dtype(dff[c])]
    feature_pool = diff_cols + base_cols

    if feature_pool:
        method = st.radio("Correlation method", ["Pearson", "Spearman"], horizontal=True, index=0)
        method_key = "pearson" if method == "Pearson" else "spearman"
        top_n = st.slider("Show top N features", min_value=5, max_value=50, value=20, step=5)

        corr_df = corr_with_target(dff, features=feature_pool, method=method_key).head(top_n)
        plot_df = corr_df.iloc[::-1]

        fig = plt.figure(figsize=(10, max(4, 0.35 * len(plot_df))))
        plt.barh(plot_df["feature"], plot_df["corr"])
        plt.axvline(0, linewidth=1)
        plt.xlabel(f"{method} correlation with home_win_binary")
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(corr_df, use_container_width=True)
    else:
        st.info("No numeric features available for correlation under current filters.")

    st.subheader("Heatmap (user-selected variables)")
    heat_pool = diff_cols + base_cols
    if heat_pool:
        default_heat = [c for c in heat_pool if any(k in c for k in ["shots", "on_target", "fouls", "yellow", "possession", "chances", "corners"])]
        default_heat = default_heat[:12] if default_heat else heat_pool[:12]

        heat_sel = st.multiselect("Variables for heatmap", heat_pool, default=default_heat)
        cmap_name = st.selectbox("Colormap", ["RdBu_r", "coolwarm", "PiYG", "viridis"], index=0)
        center_zero = st.checkbox("Center colors at 0 (recommended)", value=True)

        if len(heat_sel) >= 3:
            m = dff[heat_sel].corr(numeric_only=True)

            fig2 = plt.figure(figsize=(10, 8))
            if center_zero:
                vmax = np.nanmax(np.abs(m.values))
                vmin, vmax = (-vmax, vmax) if np.isfinite(vmax) else (-1, 1)
                plt.imshow(m.values, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)
            else:
                plt.imshow(m.values, aspect="auto", cmap=cmap_name)

            plt.colorbar(label="Correlation")
            plt.xticks(range(len(heat_sel)), heat_sel, rotation=75, ha="right")
            plt.yticks(range(len(heat_sel)), heat_sel)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("Select at least 3 variables for the heatmap.")
    else:
        st.info("No variables available for the heatmap under current filters.")

    st.caption("Correlations show association, not causation. Many match stats reflect game state.")

# -------------------------
# Tab 3: Attendance & Possession
# -------------------------
with tab3:
    st.subheader("Attendance & Possession Checks (Association)")

    include_closed_doors = st.checkbox(
        "Include closed-door matches (attendance = 0) in attendance boxplot",
        value=False,
        key="tab3_include_closed_doors"
    )

    att_df = dff.copy()
    if "attendance" in att_df.columns:
        att_df["attendance"] = pd.to_numeric(att_df["attendance"], errors="coerce")
        if not include_closed_doors:
            att_df = att_df[att_df["attendance"] > 0]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Attendance vs Home Win (boxplot)**")
        if {"attendance", "home_win_binary"}.issubset(att_df.columns):
            win = att_df.loc[att_df["home_win_binary"] == 1, "attendance"].dropna()
            notwin = att_df.loc[att_df["home_win_binary"] == 0, "attendance"].dropna()

            if len(win) < 2 or len(notwin) < 2:
                st.info("Not enough attendance data in both groups under current filters.")
            else:
                fig = plt.figure(figsize=(6, 4))
                plt.boxplot([win.values, notwin.values], labels=["Home Win", "Not Home Win"])
                plt.ylabel("Attendance")
                plt.title(
                    "Attendance vs Home Win"
                    + (" (including closed doors)" if include_closed_doors else " (excluding closed doors)")
                )
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("Attendance column not available or not numeric.")

    with c2:
        st.markdown("**Possession diff vs Home Win (scatter)**")
        if "possession_diff" in dff.columns and pd.api.types.is_numeric_dtype(dff["possession_diff"]):
            fig2 = plt.figure(figsize=(6, 4))
            plt.scatter(dff["possession_diff"], dff["home_win_binary"], alpha=0.25)
            plt.xlabel("Possession diff (home_possessions - away_possessions)")
            plt.ylabel("Home win (0/1)")
            plt.title("Possession diff vs Home Win")
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("possession_diff not available under current filters.")

    st.caption(
        "Note: These are associations. Attendance is confounded by team strength/popularity, "
        "and attendance = 0 reflects COVID closed-door matches."
    )

    st.markdown("### Home/Away possession vs match outcome")

    needed = {"match_result", "home_possessions", "away_possessions"}
    if needed.issubset(dff.columns):
        pos_df = dff[list(needed)].dropna().copy()

        order = ["Home Win", "Draw", "Away Win"]
        pos_df["match_result"] = pd.Categorical(
            pos_df["match_result"],
            categories=order,
            ordered=True
        )

        col3, col4 = st.columns(2)

        with col3:
            data_home = [
                pos_df.loc[pos_df["match_result"] == k, "home_possessions"].values
                for k in order
            ]

            if all(len(x) > 1 for x in data_home):
                fig3 = plt.figure(figsize=(7, 4))
                plt.boxplot(data_home, labels=order)
                plt.ylabel("Home possession")
                plt.title("Home possession vs match outcome")
                plt.tight_layout()
                st.pyplot(fig3)
            else:
                st.info("Not enough data for home possession boxplot under current filters.")

        with col4:
            data_away = [
                pos_df.loc[pos_df["match_result"] == k, "away_possessions"].values
                for k in order
            ]

            if all(len(x) > 1 for x in data_away):
                fig4 = plt.figure(figsize=(7, 4))
                plt.boxplot(data_away, labels=order)
                plt.ylabel("Away possession")
                plt.title("Away possession vs match outcome")
                plt.tight_layout()
                st.pyplot(fig4)
            else:
                st.info("Not enough data for away possession boxplot under current filters.")
    else:
        st.info("Missing columns for possession charts (need match_result, home_possessions, away_possessions).")
   

# -------------------------
# Tab 4: Association Tests
# -------------------------
with tab4:
    st.subheader("Association tests (Home win vs Not home win)")
    st.caption("Welch t-tests + Cohen’s d + bootstrap 95% CI for mean differences; FDR correction across tests.")

    candidates = []
    for f in ["attendance", "possession_diff", "shots_diff", "shots_on_target_diff", "fouls_diff", "yellow_diff", "red_diff"]:
        if f in dff.columns and pd.api.types.is_numeric_dtype(dff[f]):
            candidates.append(f)

    # Also allow any other *_diff
    diff_cols = [c for c in dff.columns if c.endswith("_diff") and pd.api.types.is_numeric_dtype(dff[c])]
    for c in diff_cols:
        if c not in candidates:
            candidates.append(c)

    if not candidates:
        st.info("No numeric features available to test under current filters.")
    else:
        n_boot = st.slider("Bootstrap resamples (CI)", 200, 5000, 2000, step=200)
        selected = st.multiselect(
            "Select features to test",
            candidates,
            default=candidates[: min(10, len(candidates))],
        )

        if selected:
            rows = []
            for f in selected:
                # For attendance: default exclude closed doors in the test (to avoid 0-dominance)
                if f == "attendance" and "attendance" in dff.columns:
                    tmp = dff.copy()
                    tmp = tmp[tmp["attendance"] > 0] if tmp["attendance"].notna().any() else tmp
                    rows.append(compare_groups(tmp, f, n_boot=n_boot))
                else:
                    rows.append(compare_groups(dff, f, n_boot=n_boot))

            res = pd.DataFrame(rows)
            res["p_fdr_bh"] = benjamini_hochberg(res["welch_t_pvalue"].values)
            res["sig_fdr_0_05"] = (res["p_fdr_bh"] <= 0.05)
            res = res.sort_values(["p_fdr_bh", "cohens_d"], ascending=[True, False])

            st.dataframe(res, use_container_width=True)

            st.subheader("Mean difference (Home win - Not home win) with 95% CI")
            plot_k = st.slider("How many to plot", 3, min(20, len(res)), min(10, len(res)))
            plot_res = res.head(plot_k).iloc[::-1]

            fig = plt.figure(figsize=(10, max(4, 0.45 * len(plot_res))))
            y = np.arange(len(plot_res))
            means = plot_res["mean_diff"].values
            lo = plot_res["ci_low"].values
            hi = plot_res["ci_high"].values
            err_left = means - lo
            err_right = hi - means

            plt.errorbar(means, y, xerr=[err_left, err_right], fmt="o")
            plt.axvline(0, linewidth=1)
            plt.yticks(y, plot_res["feature"])
            plt.xlabel("Mean difference with 95% CI (Home win - Not home win)")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Select at least one feature.")

# -------------------------
# Tab 5: Trends & Binning
# -------------------------
with tab5:
    st.subheader("Trends & binned relationships")

    if "year" in dff.columns:
        grp = dff.groupby("year", dropna=True)["home_win_binary"].mean().reset_index().sort_values("year")
        fig = plt.figure(figsize=(10, 4))
        plt.plot(grp["year"], grp["home_win_binary"], marker="o")
        plt.ylim(0, 1)
        plt.xlabel("Year")
        plt.ylabel("Home win rate")
        plt.tight_layout()
        st.pyplot(fig)

        if "attendance" in dff.columns and pd.api.types.is_numeric_dtype(dff["attendance"]):
            grp2 = dff.groupby("year", dropna=True)["attendance"].mean().reset_index().sort_values("year")
            fig2 = plt.figure(figsize=(10, 4))
            plt.plot(grp2["year"], grp2["attendance"], marker="o")
            plt.xlabel("Year")
            plt.ylabel("Avg attendance")
            plt.tight_layout()
            st.pyplot(fig2)

    st.markdown("### Binned win-rate (quantiles)")
    bin_candidates = []
    for c in ["attendance", "possession_diff", "shots_diff", "shots_on_target_diff"]:
        if c in dff.columns and pd.api.types.is_numeric_dtype(dff[c]):
            bin_candidates.append(c)

    # allow other diffs too
    for c in [c for c in dff.columns if c.endswith("_diff") and pd.api.types.is_numeric_dtype(dff[c])]:
        if c not in bin_candidates:
            bin_candidates.append(c)

    if bin_candidates:
        xcol = st.selectbox("Variable to bin", bin_candidates, index=0)
        bins = st.slider("Number of quantile bins", 4, 20, 10)
        bdf = binned_win_rate(dff, xcol, bins=bins)
        if bdf.empty:
            st.info("Not enough distinct values to bin with the chosen settings.")
        else:
            fig = plt.figure(figsize=(10, 4))
            plt.plot(bdf["x_mean"], bdf["home_win_rate"], marker="o")
            plt.xlabel(f"{xcol} (bin mean)")
            plt.ylabel("Home win rate")
            plt.ylim(0, 1)
            plt.tight_layout()
            st.pyplot(fig)
            st.dataframe(bdf, use_container_width=True)
    else:
        st.info("No numeric variables available for binning under current filters.")

st.divider()
st.caption(
    "Reminder: dashboard outputs show association, not causation. Many match stats are in-match measures influenced by game state. "
    "Attendance is confounded by team strength/popularity and includes COVID closed-door matches (attendance=0)."
)

# -------------------------
# Tab 6: Predictive Models: Logistic Regression + Gradient Boosting
# -------------------------

with tab6:
    st.subheader("Predictive Models: Logistic Regression & Gradient Boosting")
    st.caption(
        "This tab reproduces the team’s LR/GB approach using the selected feature set "
        "(possession_diff, shots_diff, shots_on_target_diff, fouls/yellow/red diffs, attendance, attendance_high) "
        "with a time-based holdout split and TimeSeriesSplit CV."
    )

    results = fit_lr_gb_models(dff)

    if "error" in results:
        st.error(results["error"])
    else:
        st.write(f"Rows used for modeling (after dropping NA): **{results['n_rows_used']}**")

        # ---- Holdout results
        st.markdown("### Holdout (train: year < 2023, test: year ≥ 2023)")
        c1, c2 = st.columns(2)
        c1.metric("Logistic Regression Accuracy", f"{results['lr_holdout_acc']:.3f}")
        c2.metric("Gradient Boosting Accuracy", f"{results['gb_holdout_acc']:.3f}")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**LR Confusion Matrix**")
            st.dataframe(pd.DataFrame(results["lr_holdout_cm"], index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))
        with c4:
            st.markdown("**GB Confusion Matrix**")
            st.dataframe(pd.DataFrame(results["gb_holdout_cm"], index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

        # ---- Feature importance (holdout fit)
        st.markdown("### Feature importance (holdout fit)")
        c5, c6 = st.columns(2)

        with c5:
            st.markdown("**Logistic Regression (|coefficient|)**")
            df_lr = results["lr_holdout_coef"].head(10).iloc[::-1]
            fig = plt.figure(figsize=(8, 4))
            plt.barh(df_lr["Feature"], df_lr["AbsCoeff"])
            plt.xlabel("|Coefficient|")
            plt.tight_layout()
            st.pyplot(fig)

        with c6:
            st.markdown("**Gradient Boosting (importance)**")
            df_gb = results["gb_holdout_imp"].head(10).iloc[::-1]
            fig = plt.figure(figsize=(8, 4))
            plt.barh(df_gb["Feature"], df_gb["Importance"])
            plt.xlabel("Importance")
            plt.tight_layout()
            st.pyplot(fig)

        # ---- Time-based CV results
        st.markdown("### Time-based cross-validation (TimeSeriesSplit, 4 folds)")
        c7, c8 = st.columns(2)
        c7.metric("LR CV Mean Accuracy", f"{results['lr_cv_mean_acc']:.3f}")
        c8.metric("GB CV Mean Accuracy", f"{results['gb_cv_mean_acc']:.3f}")

        # Fold accuracy lines
        fig = plt.figure(figsize=(8, 4))
        folds = np.arange(1, 5)
        plt.plot(folds, results["lr_cv_fold_acc"], marker="o", label="LR")
        plt.plot(folds, results["gb_cv_fold_acc"], marker="o", label="GB")
        plt.ylim(0, 1)
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.title("CV Accuracy per Fold")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # CV confusion matrices
        c9, c10 = st.columns(2)
        with c9:
            st.markdown("**LR CV Confusion Matrix (aggregated)**")
            st.dataframe(pd.DataFrame(results["lr_cv_cm"], index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))
        with c10:
            st.markdown("**GB CV Confusion Matrix (aggregated)**")
            st.dataframe(pd.DataFrame(results["gb_cv_cm"], index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

        # CV importances
        st.markdown("### Average importance across CV folds")
        c11, c12 = st.columns(2)
        with c11:
            st.markdown("**LR Average |coefficient| across folds**")
            lr_cv = results["lr_cv_coef"].copy()
            lr_cv["AbsCoeff"] = lr_cv["AvgCoefficient"].abs()
            lr_cv = lr_cv.sort_values("AbsCoeff", ascending=False).head(10).iloc[::-1]
            fig = plt.figure(figsize=(8, 4))
            plt.barh(lr_cv["Feature"], lr_cv["AbsCoeff"])
            plt.xlabel("Avg |Coefficient|")
            plt.tight_layout()
            st.pyplot(fig)

        with c12:
            st.markdown("**GB Average importance across folds**")
            gb_cv = results["gb_cv_imp"].head(10).iloc[::-1]
            fig = plt.figure(figsize=(8, 4))
            plt.barh(gb_cv["Feature"], gb_cv["AvgImportance"])
            plt.xlabel("Avg Importance")
            plt.tight_layout()
            st.pyplot(fig)

        st.caption(
            "Note: Very high accuracies can occur if leakage features are included (e.g., goal-based fields or engineered fields tightly tied to the outcome). "
            "This tab uses the team’s selected, more defensible feature set for comparison."
        )