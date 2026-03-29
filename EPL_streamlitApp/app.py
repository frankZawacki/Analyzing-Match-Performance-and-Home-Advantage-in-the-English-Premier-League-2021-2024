# NOTES TO RUN THIS APP LOCALLY:
# app.py
# EPL Match Performance & Home Advantage Dashboard (tabbed version + COVID attendance handling)
# Data file: mydata.csv (place in the SAME directory as this app.py)
#
# Run:
#   pip install pandas numpy matplotlib scipy statsmodels streamlit
#   streamlit run app.py

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
import statsmodels.api as sm

# -------------------------
# Config
# -------------------------
CSV_PATH = "mydata.csv"

st.set_page_config(page_title="EPL Home Advantage Dashboard", layout="wide")


# -------------------------
# Helpers: cleaning & features
# -------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")
        for c in df.columns
    ]
    return df


def parse_mixed_epl_dates(date_series: pd.Series) -> pd.Series:
    """
    Handles mixed formats like:
      - "13th August 2021"
      - "23/05/2021" (UK day/month/year)
    """
    s = date_series.astype(str).str.strip()

    # Remove ordinal suffixes: 13th -> 13, 1st -> 1, etc.
    s_no_ord = s.str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True)

    # Identify slash dates like 23/05/2021
    mask_slash = s_no_ord.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$", na=False)

    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # Parse slash dates explicitly as DAY/MONTH/YEAR
    dt.loc[mask_slash] = pd.to_datetime(
        s_no_ord.loc[mask_slash],
        format="%d/%m/%Y",
        errors="coerce",
    )

    # Parse the rest using dayfirst=True (safe for UK-style)
    dt.loc[~mask_slash] = pd.to_datetime(
        s_no_ord.loc[~mask_slash],
        errors="coerce",
        dayfirst=True,
    )

    return dt


def clean_epl_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans your Kaggle EPL dataset schema:
      - attendance like '"60,095"' -> numeric
      - date in mixed formats -> date_dt
      - class outcome h/d/a -> outcome_hda + home_win_binary
      - builds season_label and calendar_year
      - builds closed_doors flag for COVID period (attendance==0)
    """
    df = normalize_columns(df)

    # Attendance: '"60,095"' -> 60095
    if "attendance" in df.columns:
        df["attendance"] = (
            df["attendance"].astype(str)
            .str.replace('"', "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")

    # Date parsing (mixed formats)
    if "date" in df.columns:
        df["date_dt"] = parse_mixed_epl_dates(df["date"])
        df["calendar_year"] = df["date_dt"].dt.year.astype("Int64")

        # EPL season starts around Aug
        season_start_year = np.where(
            df["date_dt"].dt.month >= 8,
            df["date_dt"].dt.year,
            df["date_dt"].dt.year - 1,
        )
        df["season_start"] = pd.Series(season_start_year, index=df.index).astype("Int64")
        df["season_end"] = (df["season_start"] + 1).astype("Int64")
        df["season_label"] = df["season_start"].astype(str) + "-" + df["season_end"].astype(str).str[-2:]

    # Goals
    for c in ["goals_home", "away_goals"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "goals_home" in df.columns and "home_goals" not in df.columns:
        df["home_goals"] = df["goals_home"]

    # Outcome from class: h/d/a
    if "class" in df.columns:
        df["outcome_hda"] = df["class"].astype(str).str.lower().map({"h": "H", "d": "D", "a": "A"})
    else:
        # Fallback from goals if class missing
        df["outcome_hda"] = np.where(
            df["home_goals"] > df["away_goals"],
            "H",
            np.where(df["home_goals"] == df["away_goals"], "D", "A"),
        )

    df["home_win_binary"] = (df["outcome_hda"] == "H").astype(int)
    df["is_draw"] = (df["outcome_hda"] == "D").astype(int)

    # Derived columns
    if "home_goals" in df.columns and "away_goals" in df.columns:
        df["goal_diff"] = df["home_goals"] - df["away_goals"]

    if "home_possessions" in df.columns and "away_possessions" in df.columns:
        df["home_possessions"] = pd.to_numeric(df["home_possessions"], errors="coerce")
        df["away_possessions"] = pd.to_numeric(df["away_possessions"], errors="coerce")
        df["possession_diff"] = df["home_possessions"] - df["away_possessions"]

    # Treat team columns as strings (your sample shows numeric codes)
    for c in ["home_team", "away_team"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # COVID / closed doors indicator (attendance==0)
    if "attendance" in df.columns:
        df["closed_doors"] = df["attendance"].fillna(0).eq(0).astype(int)
    else:
        df["closed_doors"] = 0

    return df


def build_home_away_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For any numeric pair home_X and away_X, create X_diff = home_X - away_X
    """
    df = df.copy()
    cols = df.columns
    home_cols = [c for c in cols if c.startswith("home_")]

    for hc in home_cols:
        base = hc.replace("home_", "", 1)
        ac = "away_" + base
        if ac in cols and pd.api.types.is_numeric_dtype(df[hc]) and pd.api.types.is_numeric_dtype(df[ac]):
            diff_name = f"{base}_diff"
            if diff_name not in df.columns:
                df[diff_name] = df[hc] - df[ac]
    return df


def corr_with_target(df: pd.DataFrame, target="home_win_binary", features=None, method="pearson") -> pd.DataFrame:
    """
    Correlation with binary target (association).
    method: 'pearson' or 'spearman'
    """
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
    """
    Bootstrap CI for difference in means: mean(x) - mean(y)
    """
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
    """
    Welch t-test + Cohen's d + bootstrap CI for mean difference (home win - not home win)
    """
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
    """
    Benjamini–Hochberg FDR correction. Returns adjusted p-values.
    """
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
    """
    Bin xcol into quantiles and compute home win rate per bin.
    """
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
def load_data(path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(path)
    df = clean_epl_dataset(df_raw)
    df = build_home_away_diffs(df)
    return df


# -------------------------
# Load
# -------------------------
st.title("EPL Match Performance & Home Advantage (Interactive)")
st.caption("Tabs include: home advantage, correlations/heatmap, attendance/possession checks, trends, and interpretable stats.")

try:
    df = load_data(CSV_PATH)
except FileNotFoundError:
    st.error(f"Could not find '{CSV_PATH}'. Put mydata.csv in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")

dff = df.copy()

filter_mode = st.sidebar.radio("Filter by", ["Calendar year", "Season (label)"], index=0)

if filter_mode == "Calendar year":
    if "calendar_year" in dff.columns:
        years = sorted([int(y) for y in dff["calendar_year"].dropna().unique()])
        year_sel = st.sidebar.multiselect("Year", years, default=years)
        if year_sel:
            dff = dff[dff["calendar_year"].isin(year_sel)]
    else:
        st.sidebar.info("calendar_year not available (date parsing may have failed).")
else:
    if "season_label" in dff.columns:
        seasons = sorted([s for s in dff["season_label"].dropna().unique()])
        season_sel = st.sidebar.multiselect("Season", seasons, default=seasons)
        if season_sel:
            dff = dff[dff["season_label"].isin(season_sel)]
    else:
        st.sidebar.info("season_label not available (date parsing may have failed).")

if "home_team" in dff.columns and "away_team" in dff.columns:
    teams = sorted(pd.unique(pd.concat([dff["home_team"], dff["away_team"]], ignore_index=True)))
    team_sel = st.sidebar.selectbox("Team (optional)", ["(All)"] + teams)
    if team_sel != "(All)":
        dff = dff[(dff["home_team"] == team_sel) | (dff["away_team"] == team_sel)]
else:
    team_sel = "(All)"

outcome_sel = st.sidebar.multiselect("Outcome (H/D/A)", ["H", "D", "A"], default=["H", "D", "A"])
if outcome_sel:
    dff = dff[dff["outcome_hda"].isin(outcome_sel)]

st.sidebar.divider()
st.sidebar.caption("Tip: If charts look sparse, widen your filters (all years/seasons/teams).")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Home Advantage", "Correlations & Heatmap", "Attendance & Possession", "Association Tests", "Trends & Binning"]
)

# -------------------------
# Tab 1: Home Advantage
# -------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(dff):,}")
    c2.metric("Home Win Rate", f"{dff['home_win_binary'].mean():.3f}")
    c3.metric("Draw Rate", f"{dff['is_draw'].mean():.3f}")
    if "goal_diff" in dff.columns:
        c4.metric("Avg Goal Diff (Home-Away)", f"{dff['goal_diff'].mean():.3f}")
    else:
        c4.metric("Avg Goal Diff (Home-Away)", "N/A")

    st.subheader("Goal Difference Distribution (Home - Away)")
    if "goal_diff" in dff.columns:
        fig = plt.figure(figsize=(10, 4))
        plt.hist(dff["goal_diff"].dropna(), bins=20)
        plt.axvline(0, linewidth=1)
        plt.xlabel("Goal difference (home_goals - away_goals)")
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Outcome Rates (H/D/A)")
    rates = dff["outcome_hda"].value_counts(normalize=True).reindex(["H", "D", "A"]).fillna(0)
    fig2 = plt.figure(figsize=(6, 4))
    plt.bar(rates.index, rates.values)
    plt.ylim(0, 1)
    plt.ylabel("Proportion")
    plt.tight_layout()
    st.pyplot(fig2)

    st.caption("Home advantage visuals are descriptive (association, not causation).")

# -------------------------
# Tab 2: Correlations + Heatmap
# -------------------------
with tab2:
    st.subheader("Correlations with home_win_binary (association)")
    diff_cols = [c for c in dff.columns if c.endswith("_diff") and pd.api.types.is_numeric_dtype(dff[c])]
    base_cols = []
    for c in ["attendance", "home_possessions", "away_possessions", "possession_diff"]:
        if c in dff.columns and pd.api.types.is_numeric_dtype(dff[c]):
            base_cols.append(c)

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

    st.subheader("Heatmap (user-selected variables)")
    heat_pool = diff_cols + base_cols
    if heat_pool:
        default_heat = [c for c in heat_pool if any(k in c for k in ["shots", "on", "fouls", "yellow", "corners", "possession", "pass", "chances"])]
        default_heat = default_heat[:12]
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

    st.caption("Correlations show association, not causation. Match stats may reflect game state.")

# -------------------------
# Tab 3: Attendance & Possession (adds the two requested plots + COVID handling)
# -------------------------
with tab3:
    st.subheader("Attendance & Possession Checks (Association)")

    # COVID / closed-doors handling toggle (attendance==0)
    exclude_closed = st.checkbox(
        "Exclude closed-door matches (attendance = 0) for attendance analyses",
        value=True
    )

    att_df = dff.copy()
    if "attendance" in att_df.columns and exclude_closed:
        att_df = att_df[att_df["attendance"] > 0]

    c1, c2 = st.columns(2)

    # --- Box/whisker plot: Attendance vs Home Win
    with c1:
        st.markdown("**Attendance vs Home Win (box/whisker)**")
        if "attendance" in att_df.columns and pd.api.types.is_numeric_dtype(att_df["attendance"]):
            win = att_df.loc[att_df["home_win_binary"] == 1, "attendance"].dropna()
            notwin = att_df.loc[att_df["home_win_binary"] == 0, "attendance"].dropna()

            if len(win) < 2 or len(notwin) < 2:
                st.info("Not enough attendance data in both groups (Home Win vs Not Home Win) under current filters.")
            else:
                fig = plt.figure(figsize=(6, 4))
                plt.boxplot([win.values, notwin.values], labels=["Home Win", "Not Home Win"])
                plt.ylabel("Attendance")
                plt.title("Attendance vs Home Win")
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("Attendance column not available or not numeric.")

    # --- Scatter: Possession diff vs Home Win
    with c2:
        st.markdown("**Possession diff vs Home Win (scatter)**")
        if "possession_diff" in dff.columns and pd.api.types.is_numeric_dtype(dff["possession_diff"]):
            fig = plt.figure(figsize=(6, 4))
            plt.scatter(dff["possession_diff"], dff["home_win_binary"], alpha=0.25)
            plt.xlabel("Possession diff (Home - Away)")
            plt.ylabel("Home win (0/1)")
            plt.title("Possession diff vs Home Win")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Possession diff not available (needs home_possessions and away_possessions).")

    st.caption(
        "Note: These are associations. In-match stats can be influenced by game state. "
        "Attendance can be confounded by team strength, and attendance=0 reflects COVID closed-door matches."
    )

# -------------------------
# Tab 4: Association Tests (with CI + FDR)
# -------------------------
with tab4:
    st.subheader("Association tests (Home win vs Not home win)")
    st.caption("Welch t-tests + Cohen’s d + bootstrap 95% CI for mean differences. Includes FDR correction across many tests.")

    candidates = []
    for f in ["attendance", "possession_diff"]:
        if f in dff.columns and pd.api.types.is_numeric_dtype(dff[f]):
            candidates.append(f)

    diff_cols = [c for c in dff.columns if c.endswith("_diff") and pd.api.types.is_numeric_dtype(dff[c])]
    if diff_cols:
        corr_top = corr_with_target(dff, features=diff_cols, method="pearson").head(20)["feature"].tolist()
        candidates += corr_top

    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        st.info("No numeric features available to test.")
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
                # If testing attendance, allow optional exclusion of closed-door matches (to avoid 0-dominance)
                if f == "attendance" and "attendance" in dff.columns:
                    tmp = dff.copy()
                    # mirror the choice from Tab 3 (default behavior):
                    # exclude closed doors when there are many zeros
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

            st.caption("Interpretation: significant differences can still be non-causal (game state + confounding).")
        else:
            st.info("Select at least one feature.")

# -------------------------
# Tab 5: Trends & Binning
# -------------------------
with tab5:
    st.subheader("Trends & binned relationships")

    if filter_mode == "Calendar year" and "calendar_year" in dff.columns:
        year_col = "calendar_year"
        xlabel = "Calendar year"
    elif "season_start" in dff.columns:
        year_col = "season_start"
        xlabel = "Season start year"
    else:
        year_col = None
        xlabel = ""

    if year_col is not None:
        grp = dff.groupby(year_col, dropna=True)["home_win_binary"].mean().reset_index().sort_values(year_col)
        fig = plt.figure(figsize=(10, 4))
        plt.plot(grp[year_col], grp["home_win_binary"], marker="o")
        plt.ylim(0, 1)
        plt.xlabel(xlabel)
        plt.ylabel("Home win rate")
        plt.tight_layout()
        st.pyplot(fig)

        if "attendance" in dff.columns and pd.api.types.is_numeric_dtype(dff["attendance"]):
            grp3 = dff.groupby(year_col, dropna=True)["attendance"].mean().reset_index().sort_values(year_col)
            fig3 = plt.figure(figsize=(10, 4))
            plt.plot(grp3[year_col], grp3["attendance"], marker="o")
            plt.xlabel(xlabel)
            plt.ylabel("Avg attendance")
            plt.tight_layout()
            st.pyplot(fig3)

    st.markdown("### Binned win-rate")
    bin_candidates = []
    for c in ["attendance", "possession_diff"]:
        if c in dff.columns and pd.api.types.is_numeric_dtype(dff[c]):
            bin_candidates.append(c)

    common_diff = [c for c in dff.columns if c.endswith("_diff") and any(k in c for k in ["shots", "shots_on", "fouls", "yellow", "corners", "pass", "chances"])]
    bin_candidates += common_diff[:10]
    bin_candidates = list(dict.fromkeys(bin_candidates))

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

st.divider()
st.caption(
    "Reminder: dashboard outputs show association, not causation. "
    "In-match stats can reflect game state, and attendance can be confounded by team strength/popularity. "
    "Attendance=0 indicates closed-door matches during COVID."
)