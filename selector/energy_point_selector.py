"""Energy point selector using RFE regression."""

import numpy as np

from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, RepeatedKFold


class Selector:
    """Select the features that most describe the reference standards."""

    def __init__(self, data, coeffs):
        """
        Init function.

        Attributes:
            Data - Spectra that are made of linear combinations of
                reference spectra.
            Coefficients - The coefficients of each reference that
                generated the dataset.  Must add to one.
        """
        self.Data = data
        self.Coeffs = coeffs

    def select_energy_points(self, n_points=None, estimator='dt', verbose=True,
                             scoring='neg_root_mean_squared_error',
                             **kwargs):
        """Return the n_points that are most important."""
        if n_points is None:
            n_points = len(self.Data)

        if estimator.lower().replace(' ', '') in ['linear',
                                                  'linearregression']:
            model = LinearRegression(**kwargs)

        elif estimator.lower().replace(' ', '') in ['dt', 'decisiontree']:
            model = DecisionTreeRegressor(**kwargs)

        elif estimator.lower().replace(' ', '') in ['rf', 'randomforest']:
            model = RandomForestRegressor(**kwargs)

        else:
            print("Estimator model not supported. " +
                  "Setting to default Decision Tree Regressor.")
            model = DecisionTreeRegressor(**kwargs)

        rfe = RFE(model, n_features_to_select=n_points, step=1)
        self.rfe = rfe.fit(self.Data, self.Coeffs)

        if verbose:
            return self.rfe, self.evaluate_rfe(scoring=scoring)
        else:
            return self.rfe

    def evaluate_rfe(self, verbose=True,
                     scoring='neg_root_mean_squared_error'):
        """Evaluate model using cross validation."""
        cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
        n_scores = cross_val_score(self.rfe.estimator_, self.Data,
                                   self.Coeffs, cv=cv, scoring=scoring)
        print('Score: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        return np.mean(n_scores)
