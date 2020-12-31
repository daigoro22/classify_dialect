#!/bin/sh
python optuna_optimize_cb.py -d all_param_early_stopping
python confirm_validity_of_optuna_params.py -d all_param_early_stopping -rd all_param_early_stopping -s optuna/optuna_all_param_early_stopping.db