#!/bin/sh
python confirm_validity_of_optuna_params.py -d best -rd best -pts best
python confirm_validity_of_optuna_params.py -d gc -rd grad_clipping -pts grad_clipping
python confirm_validity_of_optuna_params.py -d lr -rd learning_rate -pts learning_rate
python confirm_validity_of_optuna_params.py -d beta -rd beta_confirm -pts beta
python confirm_validity_of_optuna_params.py -d lstm -rd lstm -pts n_lstm