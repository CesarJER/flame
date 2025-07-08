#! -*- coding: utf-8 -*-

# Description    Flame feature selection methods
##
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
# Copyright 2018 Manuel Pastor
##
# This file is part of Flame
##
# Flame is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3.
##
# Flame is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with Flame.  If not, see <http://www.gnu.org/licenses/>.

""" This file contains implemented methods to perform
    feature selection"""

# from sklearn.preprocessing import MinMaxScaler 
# from sklearn.feature_selection import chi2
from sklearn.feature_selection import  SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
# from flame.util import get_logger
import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

def run_feature_selection(X, Y, method, num_features, quantitative):
    """
    Feature selection supporting:
    - SelectKBest (f_classif, f_regression)
    - RFE (rfe_log, rfe_rf)
    - Direct mask input (0/1 or boolean array)

    Parameters:
    - X: numpy array [n_samples, n_features]
    - Y: numpy array [n_samples]
    - method: str or array/list of 0/1 or booleans
    - num_features: int or "auto"
    - quantitative: bool

    Returns:
    - success: True/False
    - mask: boolean array of selected features (or Exception)
    """

    nobj, nvarx = np.shape(X)
    variable_mask = np.zeros(nvarx, dtype=bool)

    # --- Direct mask input (0/1 or bool) ---
    if isinstance(method, (list, np.ndarray)):
        arr = np.array(method)
        if arr.dtype in [np.bool_, bool, np.int_, int, np.uint8] and arr.shape[0] == nvarx:
            variable_mask = arr.astype(bool)
            return True, variable_mask
        else:
            return False, ValueError("Invalid mask input: must be list/array of 0/1 or bool, length = number of variables")

    # --- Determine number of features ---
    # When auto, the 10% top informative variables are retained.
    if num_features in [None, '']:
        num_features = 'auto'

    if num_features == "auto":
        # Use 10% of the total number of objects:
        # The number of variables is greater than the 10% of the objects
        # And the number of objects is greater than 100
        if nvarx > (nobj * 0.1) and nobj >= 100:
            n_features = int(nobj * 0.1)
        # If number of objects is smaller than 100 then n_features
        # is set to 10
        elif nobj < 100:
            n_features = 10
        # In any other circunstance set number of variables to 10 
        else:
            n_features = nvarx
    else:
        try:
            n_features = int(num_features)
        except:
            n_features = nvarx
        if n_features > nvarx or n_features < 1:
            n_features = nvarx

    # --- RFE Selection ---
    if method.lower() in ["rfe_log", "rfe_rf"]:
        try:
            if quantitative:
                estimator = LinearRegression() if method == "rfe_log" else RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                estimator = LogisticRegression(solver='liblinear', max_iter=1000) if method == "rfe_log" else RandomForestClassifier(n_estimators=100, random_state=42)

            selector = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
            selector.fit(X, Y)
            variable_mask = selector.get_support()
            return True, variable_mask

        except Exception as e:
            return False, e

    # --- SelectKBest fallback ---
    function = f_classif if not quantitative else f_regression

    try:
        kbest = SelectKBest(score_func=function, k=n_features)
        kbest.fit(X,Y)
        variable_mask = kbest.get_support()
        return True, variable_mask
    except Exception as e:
        return False, e

