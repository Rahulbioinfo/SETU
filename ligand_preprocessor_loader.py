import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class LigandPreprocessor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        X = X.copy()

        self.columns_ = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        Z = X[self.columns_].replace(
            [np.inf, -np.inf],
            np.nan
        )

        self.imputer_ = SimpleImputer(
            strategy="median"
        )

        Z = self.imputer_.fit_transform(Z)

        self.scaler_ = StandardScaler()
        self.scaler_.fit(Z)

        return self

    def transform(self, X):

        X = X.copy()

        Z = X[self.columns_].replace(
            [np.inf, -np.inf],
            np.nan
        )

        Z = self.imputer_.transform(Z)
        Z = self.scaler_.transform(Z)

        return Z.astype(np.float32)

    def get_feature_names_out(
        self,
        input_features=None
    ):

        return np.asarray(
            self.columns_,
            dtype=object
        )
