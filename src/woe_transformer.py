import pandas as pd
import numpy as np
from scripts.constants import WOE_CANDIDATE_COLS, TARGET_COL


class WoeTransformer:
    """
    A class to perform Weight of Evidence (WoE) transformation on a dataset.
    Attributes:
        df (pd.DataFrame): The input dataframe to be transformed.
        transformed_df (pd.DataFrame): The dataframe after binning and category merging.
        bins (int): Number of bins for numeric features.
        category_count_min_threashold (int): Minimum count threshold for categorical features.
        EPS (float): Smoothing constant to avoid division by zero.
        woe_maps (dict): A dictionary to store WoE mappings for each feature.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.transformed_df = None
        self.bins = 10
        self.category_count_min_threashold = 30
        self.EPS = 0.5  # Smoothing constant
        self.woe_maps = {}
        self.binned_df = None

        # Learned during fit
        self.numeric_bin_edges = {}  # feature -> bin edges
        self.category_merge_map = {}  # feature -> set(low volume categories)

    # =========================
    # FIT PHASE
    # =========================
    def _fit_numeric(self, feature: pd.Series) -> pd.Series:
        _, bin_edges = pd.qcut(feature, q=self.bins, duplicates="drop", retbins=True)

        # Extend edges to handle out-of-range values
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        self.numeric_bin_edges[feature.name] = bin_edges

    def _fit_categorical(self, feature: pd.Series):
        feature_name = feature.name

        # Initialize set for this feature if not exists
        if feature_name not in self.category_merge_map:
            self.category_merge_map[feature_name] = set()

        feature_counts = feature.value_counts()

        for category, count in feature_counts.items():
            if (
                category != "transport"  # protected category
                and count <= self.category_count_min_threashold
            ):
                self.category_merge_map[feature_name].add(category)

    # =========================
    # TRANSFORM PHASE
    # =========================
    def _transform_numeric(self, feature: pd.Series) -> pd.Series:
        bin_edges = self.numeric_bin_edges[feature.name]
        bin_labels = [f"bin_{i}" for i in range(len(bin_edges) - 1)]

        binned = pd.cut(feature, bins=bin_edges, labels=bin_labels, include_lowest=True)
        return binned.astype("object").fillna("MISSING")

    def _transform_categorical(self, feature: pd.Series) -> pd.Series:
        feature_name = feature.name
        low_volume_categories = self.category_merge_map.get(feature_name, set())
        return feature.apply(
            lambda x: "OTHER_LOW_VOLUME" if x in low_volume_categories else x
        )

    def _fit(self, feature_cols):
        for col in feature_cols:
            series = self.df[col]

            if series.dtype == "object":
                self._fit_categorical(series)
            else:
                self._fit_numeric(series)

        return self

    def _transform(self, feature_cols):
        working_df = self.df.copy()

        for col in feature_cols:
            if working_df[col].dtype == "object":
                working_df[col] = self._transform_categorical(working_df[col])
            else:
                working_df[col] = self._transform_numeric(working_df[col])

        self.binned_df = working_df
        return working_df

    # =========================
    # PUBLIC METHODS
    # =========================
    def fit_transform(self):
        self._fit(WOE_CANDIDATE_COLS)
        self.transformed_df = self._transform(WOE_CANDIDATE_COLS)
        return self.transformed_df

    def get_iv_table(self):
        working_df = self.transformed_df.copy()
        feature_cols = [c for c in working_df.columns if c != TARGET_COL]
        iv_results = []
        total_good = (working_df[TARGET_COL] == 0).sum()
        total_bad = (working_df[TARGET_COL] == 1).sum()
        for col in feature_cols:
            tmp = pd.DataFrame(
                {"bin": working_df[col], TARGET_COL: working_df[TARGET_COL]}
            )

            grouped = tmp.groupby("bin")[TARGET_COL].agg(total="count", bad="sum")

            grouped["good"] = grouped["total"] - grouped["bad"]

            # Apply smoothing
            grouped["good_s"] = grouped["good"] + self.EPS
            grouped["bad_s"] = grouped["bad"] + self.EPS

            grouped["dist_good"] = grouped["good_s"] / (
                total_good + self.EPS * len(grouped)
            )
            grouped["dist_bad"] = grouped["bad_s"] / (
                total_bad + self.EPS * len(grouped)
            )

            grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
            grouped["iv"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped[
                "woe"
            ]

            self.woe_maps[col] = grouped["woe"].to_dict()

            feature_iv = grouped["iv"].sum()
            iv_results.append({"feature": col, "iv": feature_iv})

        return pd.DataFrame(iv_results).sort_values("iv", ascending=False)

    # =========================
    # WOe TRANSFORM PHASE
    # =========================
    def transform_to_woe(self):
        if self.binned_df is None:
            raise ValueError("Call fit_transform() first")

        woe_df = self.binned_df.copy()

        for feature, woe_map in self.woe_maps.items():
            woe_df[feature] = woe_df[feature].map(woe_map)

        self.woe_df = woe_df
        return woe_df
