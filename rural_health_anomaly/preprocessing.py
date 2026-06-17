"""Data preprocessing and feature engineering."""

from __future__ import annotations

import ast
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from .config import PreprocessingConfig
from .schema import SCHEMA_EXCLUDED_TEXT_FEATURES


def _make_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError("scaler must be 'standard' or 'minmax'")


def _safe_std(series: pd.Series) -> float:
    value = series.std(ddof=0)
    return 0.0 if pd.isna(value) else float(value)


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    return list(OrderedDict.fromkeys(values))


def _transformation_metadata(feature_name: str) -> tuple[list[str], int, str]:
    """Infer a simple transformation path from a feature name."""

    if "__" in feature_name:
        return ["raw", "multi_value_expand"], 2, "expanded_multi_value"

    if any(
        feature_name.startswith(f"{base}_mean_")
        or feature_name.startswith(f"{base}_std_")
        or feature_name.startswith(f"{base}_lag")
        for base in (
            "heart_rate_bpm",
            "systolic_bp_mmhg",
            "diastolic_bp_mmhg",
            "spo2_percent",
            "body_temperature_c",
            "respiratory_rate_bpm",
            "weight_kg",
            "height_cm",
            "bmi_kg_m2",
            "glucose_fasting_mg_dl",
            "glucose_postprandial_mg_dl",
            "hb_g_dl",
            "wbc_count_10e9_l",
            "platelets_10e9_l",
            "hba1c_percent",
            "ldl_mg_dl",
            "hdl_mg_dl",
            "triglycerides_mg_dl",
            "alt_u_l",
            "ast_u_l",
            "bilirubin_mg_dl",
            "creatinine_mg_dl",
            "bun_mg_dl",
            "egfr_ml_min_1_73m2",
            "sodium_mmol_l",
            "potassium_mmol_l",
            "calcium_mg_dl",
            "age_years",
            "visits_last_90_days",
            "symptom_duration_days",
            "sanitation_index",
            "nutritional_score",
            "distance_to_nearest_facility_km",
            "treatment_response_score",
            "readmission_frequency",
            "drug_adherence_rate",
        )
    ):
        return ["raw", "time_series_engineer"], 2, "engineered_time_series"

    if any(
        feature_name.startswith(f"{prefix}_")
        for prefix in (
            "gender",
            "location_type",
            "source_type",
            "operator_id",
            "device_id",
            "measurement_posture",
            "data_quality_flag",
            "malaria_prevalence_level",
            "dengue_prevalence_level",
        )
    ):
        return ["raw", "one_hot_encode"], 2, "one_hot"

    return ["raw"], 1, "direct"


def _coerce_to_list(value) -> list:
    """Coerce list-like values into a Python list."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [item for item in value if item is not None and not pd.isna(item)]
    if isinstance(value, tuple) or isinstance(value, set):
        return [item for item in value if item is not None and not pd.isna(item)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.lower() in {"none", "null", "nan"}:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    return [item for item in parsed if item is not None and not pd.isna(item)]
            except Exception:
                pass
        return [part.strip() for part in text.split(",") if part.strip()]
    return [value]


def _normalize_scalar_value(value):
    """Normalize scalar missing sentinels into actual nulls."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "nan"}:
            return np.nan
        return text

    return value


def _expand_multi_value_feature(
    df: pd.DataFrame,
    column: str,
    vocabulary: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Expand one multi-value column into binary indicator columns."""

    if column not in df.columns:
        return df.copy(), list(vocabulary or [])

    data = df.copy()
    normalized = data[column].apply(_coerce_to_list)
    if vocabulary is None:
        tokens = sorted({token for values in normalized for token in values})
    else:
        tokens = list(vocabulary)

    for token in tokens:
        output_col = f"{column}__{token}"
        data[output_col] = normalized.apply(lambda values, t=token: 1.0 if t in values else 0.0)

    data = data.drop(columns=[column])
    return data, tokens


def _expand_list_numeric_feature(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert a numeric list column into summary statistics."""

    if column not in df.columns:
        return df.copy()

    data = df.copy()
    coerced = data[column].apply(_coerce_to_list)
    numeric_lists = coerced.apply(
        lambda values: [float(v) for v in values if v is not None and not pd.isna(v)]
    )

    data[f"{column}_mean"] = numeric_lists.apply(lambda values: float(np.mean(values)) if values else np.nan)
    data[f"{column}_std"] = numeric_lists.apply(lambda values: float(np.std(values, ddof=0)) if values else np.nan)
    data[f"{column}_last"] = numeric_lists.apply(lambda values: float(values[-1]) if values else np.nan)
    data[f"{column}_count"] = numeric_lists.apply(len).astype(float)
    data = data.drop(columns=[column])
    return data


def _build_group_features(
    group: pd.DataFrame,
    *,
    time_col: str,
    numeric_features: Sequence[str],
    rolling_windows_days: Sequence[int],
    lag_steps: Sequence[int],
    interaction_terms: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    """Build leakage-safe longitudinal features for one patient group."""

    group = group.sort_values(time_col).copy()
    time_indexed = group.set_index(time_col)
    feature_outputs: dict[str, pd.Series | np.ndarray] = {}

    for feature in numeric_features:
        if feature not in time_indexed.columns:
            continue

        prior_values = time_indexed[feature].shift(1)

        for window_days in rolling_windows_days:
            rolled = prior_values.rolling(f"{window_days}D", min_periods=1)
            feature_outputs[f"{feature}_mean_{window_days}d"] = rolled.mean().to_numpy()
            feature_outputs[f"{feature}_std_{window_days}d"] = rolled.apply(_safe_std, raw=False).to_numpy()

        for lag in lag_steps:
            feature_outputs[f"{feature}_lag{lag}"] = time_indexed[feature].shift(lag).to_numpy()

    for left, right in interaction_terms:
        if left in group.columns and right in group.columns:
            feature_outputs[f"{left}_x_{right}"] = group[left] * group[right]

    if feature_outputs:
        group = pd.concat([group, pd.DataFrame(feature_outputs, index=group.index)], axis=1)

    return group.reset_index(drop=True)


def engineer_longitudinal_features(
    df: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Add rolling, lag, and interaction features."""

    if df.empty:
        return df.copy()

    if config.patient_id_col not in df.columns or config.encounter_time_col not in df.columns:
        raise ValueError("patient_id_col and encounter_time_col must exist in the dataframe")

    data = df.copy()
    data[config.encounter_time_col] = pd.to_datetime(data[config.encounter_time_col], errors="coerce")
    data = data.sort_values([config.patient_id_col, config.encounter_time_col])

    engineered = (
        data.groupby(config.patient_id_col, group_keys=False)
        .apply(
            _build_group_features,
            time_col=config.encounter_time_col,
            numeric_features=config.numeric_features,
            rolling_windows_days=config.rolling_windows_days,
            lag_steps=config.lag_steps,
            interaction_terms=config.interaction_terms,
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    return engineered


class HealthcarePreprocessor:
    """End-to-end preprocessing pipeline."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.feature_pipeline_: Pipeline | None = None
        self.feature_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.multi_value_vocabularies_: dict[str, list[str]] = {}
        self.feature_provenance_: dict[str, list[str]] = {}
        self.fitted_: bool = False

    def _build_pipeline(self, numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> Pipeline:
        numeric_cols = list(numeric_cols)
        categorical_cols = list(categorical_cols)

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", KNNImputer(n_neighbors=self.config.knn_neighbors)),
                ("scaler", _make_scaler(self.config.scaler)),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_pipeline, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", categorical_pipeline, categorical_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
        steps: list[tuple[str, object]] = [("preprocessor", preprocessor)]

        if self.config.apply_pca and len(numeric_cols) + len(categorical_cols) > self.config.pca_feature_threshold:
            steps.append(("pca", PCA(n_components=self.config.pca_variance_threshold, svd_solver="full")))

        return Pipeline(steps=steps)

    def _prepare_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        data = df.copy()

        for column in data.columns:
            if column in self.config.multi_value_features or column in self.config.list_numeric_features:
                continue
            if column in {self.config.patient_id_col, self.config.encounter_time_col}:
                continue
            if pd.api.types.is_object_dtype(data[column]) or pd.api.types.is_string_dtype(data[column]):
                data[column] = data[column].apply(_normalize_scalar_value)

        for column in self.config.numeric_features:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")

        data = engineer_longitudinal_features(data, self.config)

        provenance: dict[str, list[str]] = {column: [column] for column in data.columns}

        for column in self.config.list_numeric_features:
            if column in data.columns:
                data = _expand_list_numeric_feature(data, column)
                provenance[f"{column}_mean"] = [column]
                provenance[f"{column}_std"] = [column]
                provenance[f"{column}_last"] = [column]
                provenance[f"{column}_count"] = [column]
                provenance.pop(column, None)

        for column in self.config.multi_value_features:
            vocabulary = None if fit else self.multi_value_vocabularies_.get(column, [])
            data, learned_vocab = _expand_multi_value_feature(data, column, vocabulary=vocabulary)
            if fit:
                self.multi_value_vocabularies_[column] = learned_vocab
            provenance.pop(column, None)
            for token in learned_vocab:
                provenance[f"{column}__{token}"] = [column]

        for column in list(data.columns):
            if column in provenance:
                continue

            if any(
                column.startswith(f"{base}_mean_")
                or column.startswith(f"{base}_std_")
                or column.startswith(f"{base}_lag")
                for base in self.config.numeric_features
            ):
                base_feature = next(
                    base
                    for base in self.config.numeric_features
                    if column.startswith(f"{base}_mean_")
                    or column.startswith(f"{base}_std_")
                    or column.startswith(f"{base}_lag")
                )
                provenance[column] = [self.config.patient_id_col, self.config.encounter_time_col, base_feature]
                continue

            if "_x_" in column:
                left, right = column.split("_x_", 1)
                provenance[column] = [left, right]
                continue

            provenance[column] = [column]

        self.feature_provenance_ = provenance

        return data

    def fit(self, df: pd.DataFrame, y=None) -> "HealthcarePreprocessor":
        engineered = self._prepare_features(df, fit=True)

        self.feature_columns_ = [
            c
            for c in engineered.columns
            if c not in {
                self.config.patient_id_col,
                self.config.encounter_time_col,
            }
            and c not in SCHEMA_EXCLUDED_TEXT_FEATURES
        ]

        self.numeric_columns_ = [
            c for c in self.feature_columns_ if pd.api.types.is_numeric_dtype(engineered[c])
        ]
        self.categorical_columns_ = [
            c for c in self.feature_columns_ if c not in self.numeric_columns_
        ]
        self.feature_provenance_ = {
            column: _dedupe_preserve_order(self.feature_provenance_.get(column, [column]))
            for column in self.feature_columns_
        }

        self.feature_pipeline_ = self._build_pipeline(self.numeric_columns_, self.categorical_columns_)
        self.feature_pipeline_.fit(engineered[self.feature_columns_])
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted_ or self.feature_pipeline_ is None:
            raise RuntimeError("Preprocessor must be fit before calling transform.")

        engineered = self._prepare_features(df, fit=False)
        missing_columns = [c for c in self.feature_columns_ if c not in engineered.columns]
        for column in missing_columns:
            engineered[column] = np.nan

        aligned = engineered[self.feature_columns_]
        return self.feature_pipeline_.transform(aligned)

    def fit_transform(self, df: pd.DataFrame, y=None, **fit_params) -> np.ndarray:
        self.fit(df, y=y)
        return self.transform(df)

    def get_feature_names_out(self) -> list[str]:
        if not self.fitted_ or self.feature_pipeline_ is None:
            raise RuntimeError("Preprocessor must be fit before requesting feature names.")

        if "pca" in self.feature_pipeline_.named_steps:
            pca = self.feature_pipeline_.named_steps["pca"]
            return [f"pca_{i+1}" for i in range(pca.n_components_)]

        preprocessor = self.feature_pipeline_.named_steps["preprocessor"]
        return list(preprocessor.get_feature_names_out())

    def export_feature_map(self) -> pd.DataFrame:
        """Return a compact map of final transformed columns for inspection."""

        if not self.fitted_ or self.feature_pipeline_ is None:
            raise RuntimeError("Preprocessor must be fit before exporting a feature map.")

        preprocessor = self.feature_pipeline_.named_steps["preprocessor"]
        pre_pca_feature_names = list(preprocessor.get_feature_names_out())

        if "pca" in self.feature_pipeline_.named_steps:
            pca = self.feature_pipeline_.named_steps["pca"]
            source_columns = _dedupe_preserve_order(
                source
                for feature_name in pre_pca_feature_names
                for source in self._resolve_source_columns(feature_name)
            )
            return pd.DataFrame(
                {
                    "final_feature": [f"pca_{i+1}" for i in range(pca.n_components_)],
                    "source_features": [", ".join(pre_pca_feature_names)] * pca.n_components_,
                    "source_columns": [source_columns] * pca.n_components_,
                    "transformation_path": [["raw", "scaling", "pca"]] * pca.n_components_,
                    "provenance_depth": [3] * pca.n_components_,
                    "feature_type": ["pca_component"] * pca.n_components_,
                    "n_source_features": [len(pre_pca_feature_names)] * pca.n_components_,
                }
            )

        rows: list[dict[str, object]] = []
        categorical_prefixes = sorted(self.categorical_columns_, key=len, reverse=True)

        for feature_name in pre_pca_feature_names:
            feature_type = "direct"
            source_feature = feature_name

            for prefix in categorical_prefixes:
                if feature_name.startswith(f"{prefix}_"):
                    source_feature = prefix
                    feature_type = "one_hot"
                    break

            if "__" in feature_name:
                source_feature = feature_name.split("__", 1)[0]
                feature_type = "expanded_multi_value"
            elif any(
                feature_name.startswith(f"{base}_mean_")
                or feature_name.startswith(f"{base}_std_")
                or feature_name.startswith(f"{base}_lag")
                for base in self.config.numeric_features
            ):
                feature_type = "engineered_time_series"

            transformation_path, provenance_depth, _ = _transformation_metadata(feature_name)
            rows.append(
                {
                    "final_feature": feature_name,
                    "source_features": source_feature,
                    "source_columns": self._resolve_source_columns(source_feature),
                    "transformation_path": transformation_path,
                    "provenance_depth": provenance_depth,
                    "feature_type": feature_type,
                }
            )

        return pd.DataFrame(rows)

    def _resolve_source_columns(self, feature_name: str) -> list[str]:
        """Resolve transformed features back to raw schema columns."""

        if feature_name in self.feature_provenance_:
            return _dedupe_preserve_order(self.feature_provenance_[feature_name])

        if "__" in feature_name:
            base = feature_name.split("__", 1)[0]
            return _dedupe_preserve_order(self.feature_provenance_.get(base, [base]))

        if any(
            feature_name.startswith(f"{base}_mean_")
            or feature_name.startswith(f"{base}_std_")
            or feature_name.startswith(f"{base}_lag")
            for base in self.config.numeric_features
        ):
            base = next(
                base
                for base in self.config.numeric_features
                if feature_name.startswith(f"{base}_mean_")
                or feature_name.startswith(f"{base}_std_")
                or feature_name.startswith(f"{base}_lag")
            )
            return _dedupe_preserve_order(
                [self.config.patient_id_col, self.config.encounter_time_col, base]
            )

        if feature_name in self.categorical_columns_:
            return _dedupe_preserve_order(self.feature_provenance_.get(feature_name, [feature_name]))

        if any(feature_name.startswith(f"{prefix}_") for prefix in self.categorical_columns_):
            base = next(prefix for prefix in self.categorical_columns_ if feature_name.startswith(f"{prefix}_"))
            return _dedupe_preserve_order(self.feature_provenance_.get(base, [base]))

        return _dedupe_preserve_order(self.feature_provenance_.get(feature_name, [feature_name]))

    def export_feature_map_csv(self, path: str | Path | None = None) -> str:
        """Export the feature map as CSV and optionally write it to disk."""

        feature_map = self.export_feature_map()
        csv_text = feature_map.to_csv(index=False)

        if path is not None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(csv_text, encoding="utf-8")

        return csv_text

    def export_feature_map_json(
        self,
        path: str | Path | None = None,
        *,
        orient: str = "records",
        indent: int = 2,
    ) -> str:
        """Export the feature map as JSON and optionally write it to disk."""

        feature_map = self.export_feature_map()
        json_text = feature_map.to_json(orient=orient, indent=indent)

        if path is not None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json_text, encoding="utf-8")

        return json_text
