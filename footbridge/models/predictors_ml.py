"""
Machine Learning predictor for ZSD model integration.
Extends BasePredictor to work seamlessly with existing architecture.

Usage:
    Add to PredictorFactory in predictors.py
    Train separately or as part of ModelManager
"""

from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from models.core import ModelConfig
from models.predictors import BasePredictor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


class MLPredictor(BasePredictor):
    """
    Machine Learning predictor using PPI and odds features.
    Integrates with existing ZSD prediction framework.
    """

    def __init__(
        self,
        config: ModelConfig,
        model_type: str = "gradient_boosting",
        model_path: Optional[str] = None,
    ):
        """
        Initialize ML predictor.

        Args:
            config: Model configuration
            model_type: 'logistic', 'random_forest', or 'gradient_boosting'
            model_path: Path to pre-trained model (optional)
        """
        super().__init__(config)
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False

        if model_path:
            self._load_model(model_path)
        else:
            self.model = self._create_model(model_type)

    def predict_outcome_probabilities(
        self, lambda_home: float, lambda_away: float, **kwargs
    ) -> Tuple[float, float, float]:
        """
        Predict match outcome probabilities.

        Compatible with existing predictor interface.
        Expects PPI and odds in kwargs.
        """
        if not self.is_trained:
            # Fallback to simple estimate if not trained
            return self._simple_estimate(lambda_home, lambda_away)

        # Engineer features
        features = self._engineer_features(lambda_home, lambda_away, **kwargs)

        # Scale
        features_scaled = pd.DataFrame(
            self.scaler.transform(features), columns=self.feature_names
        )

        # Predict home win probability
        prob_home = self.model.predict_proba(features_scaled)[0, 1]

        # Estimate draw and away from lambdas
        raw_diff = lambda_home - lambda_away
        prob_away = (1 - prob_home) * (1 / (1 + np.exp(raw_diff)))
        prob_draw = max(0.05, 1 - prob_home - prob_away)

        # Normalize
        total = prob_home + prob_draw + prob_away
        return (prob_home / total, prob_draw / total, prob_away / total)

    def fit(self, matches_df: pd.DataFrame) -> dict:
        """
        Train ML model on historical data.

        Args:
            matches_df: Historical matches with FTHG, FTAG, PPI, and odds

        Returns:
            Training metrics dict
        """
        # Prepare data
        data = matches_df.copy()
        data["Home_Win"] = (data["FTHG"] > data["FTAG"]).astype(int)

        # Engineer features
        features = self._engineer_all_features(data)
        self.feature_names = features.columns.tolist()

        # Split
        X = features
        y = data["Home_Win"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        # Train
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )

        from sklearn.metrics import log_loss, roc_auc_score

        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_logloss = log_loss(y_test, y_pred_proba)

        self.is_trained = True

        return {
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "test_auc": test_auc,
            "test_logloss": test_logloss,
            "n_features": len(self.feature_names),
        }

    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_package = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
        }
        joblib.dump(model_package, filepath)

    def _load_model(self, filepath: str) -> None:
        """Load pre-trained model."""
        model_package = joblib.load(filepath)
        self.model = model_package["model"]
        self.scaler = model_package["scaler"]
        self.feature_names = model_package["feature_names"]
        self.model_type = model_package.get("model_type", "unknown")
        self.is_trained = True

    def _create_model(self, model_type: str):
        """Create sklearn model."""
        models = {
            "logistic": LogisticRegression(
                random_state=42, max_iter=1000, class_weight="balanced"
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            ),
        }
        return models[model_type]

    def _engineer_features(
        self, lambda_home: float, lambda_away: float, **kwargs
    ) -> pd.DataFrame:
        """Engineer features for single prediction."""
        # Extract from kwargs
        hPPI = kwargs.get("hPPI", lambda_home)
        aPPI = kwargs.get("aPPI", lambda_away)
        PPIDiff = kwargs.get("PPIDiff", abs(hPPI - aPPI))

        psh = kwargs.get("PSH", 2.0)
        psd = kwargs.get("PSD", 3.5)
        psa = kwargs.get("PSA", 4.0)
        b365h = kwargs.get("B365H", psh)
        b365d = kwargs.get("B365D", psd)
        b365a = kwargs.get("B365A", psa)

        match_data = pd.DataFrame(
            {
                "hPPI": [hPPI],
                "aPPI": [aPPI],
                "PPIDiff": [PPIDiff],
                "PSH": [psh],
                "PSD": [psd],
                "PSA": [psa],
                "B365H": [b365h],
                "B365D": [b365d],
                "B365A": [b365a],
            }
        )

        return self._engineer_feature_set(match_data)

    def _engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for training."""
        return self._engineer_feature_set(df)

    def _engineer_feature_set(self, df: pd.DataFrame) -> pd.DataFrame:
        """Core feature engineering."""
        features = pd.DataFrame()

        # Basic PPI
        features["hPPI"] = df["hPPI"]
        features["aPPI"] = df["aPPI"]
        features["PPIDiff"] = df["PPIDiff"]

        # PPI interactions with safe division
        features["PPI_Ratio"] = df["hPPI"] / (df["aPPI"] + 0.1)
        features["PPI_Product"] = df["hPPI"] * df["aPPI"]
        features["PPI_Sum"] = df["hPPI"] + df["aPPI"]

        # Polynomial
        features["hPPI_squared"] = df["hPPI"] ** 2
        features["aPPI_squared"] = df["aPPI"] ** 2
        features["PPIDiff_squared"] = df["PPIDiff"] ** 2

        # Sharp odds with safe division (protect against 0 or very small values)
        features["Sharp_Home_Prob"] = 1 / np.maximum(df["PSH"], 1.01)
        features["Sharp_Draw_Prob"] = 1 / np.maximum(df["PSD"], 1.01)
        features["Sharp_Away_Prob"] = 1 / np.maximum(df["PSA"], 1.01)
        features["Sharp_Margin"] = (
            features["Sharp_Home_Prob"]
            + features["Sharp_Draw_Prob"]
            + features["Sharp_Away_Prob"]
            - 1
        )

        # Soft odds with safe division
        features["Soft_Home_Prob"] = 1 / np.maximum(df["B365H"], 1.01)
        features["Soft_Draw_Prob"] = 1 / np.maximum(df["B365D"], 1.01)
        features["Soft_Away_Prob"] = 1 / np.maximum(df["B365A"], 1.01)

        # Market efficiency
        features["Odds_Disagreement_Home"] = np.abs(df["PSH"] - df["B365H"])
        features["Odds_Disagreement_Away"] = np.abs(df["PSA"] - df["B365A"])

        # Combined with safe division
        ppi_ratio = df["hPPI"] / np.maximum(df["aPPI"], 0.1)
        odds_ratio = np.maximum(df["PSA"], 1.01) / np.maximum(df["PSH"], 1.01)
        features["PPI_vs_Odds"] = ppi_ratio - odds_ratio
        features["Market_Surprise"] = (df["hPPI"] - df["aPPI"]) - (
            features["Sharp_Away_Prob"] - features["Sharp_Home_Prob"]
        )

        # Quality
        features["Match_Quality"] = features["PPI_Sum"] / 2
        features["Mismatch_Indicator"] = np.abs(df["hPPI"] - df["aPPI"])

        # Clean up any remaining invalid values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())

        return features

    def _simple_estimate(
        self, lambda_home: float, lambda_away: float
    ) -> Tuple[float, float, float]:
        """Simple estimate when model not trained."""
        # Basic Poisson-like estimate
        goal_diff = lambda_home - lambda_away
        prob_home = 1 / (1 + np.exp(-goal_diff))
        prob_away = 1 - prob_home
        prob_draw = 0.25  # Rough estimate

        total = prob_home + prob_draw + prob_away
        return (prob_home / total, prob_draw / total, prob_away / total)
