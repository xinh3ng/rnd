# -*- coding: utf-8 -*-
"""Model parameters
"""
from pdb import set_trace as debug
from sklearn.model_selection import ParameterGrid

"""A grid of model parameters. Used for cross validation
"""
model_param_grid = list(ParameterGrid({
    'rank': [15],  # [15, 50]
    'max_iter': [20],  # 40 does not show improvement
    'reg_param': [0.5, 2, 5],  #
    'alpha': [1, 10]  # [0.3, 1, 1.5]
}))


"""Best set of hyperparameters
"""
best_model_params = {
    'rank': 15,
    'max_iter': 20,
    'reg_param': 2,
    'alpha': 10
}
