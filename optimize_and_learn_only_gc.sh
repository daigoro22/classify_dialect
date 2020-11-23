#!/bin/sh
python optuna_optimize_cb.py -d only_gc -s optuna/optuna_without_gc.db
python confirm_validity_of_optuna_params.py -d only_gc -rd only_gc -s optuna/optuna_only_gc.db