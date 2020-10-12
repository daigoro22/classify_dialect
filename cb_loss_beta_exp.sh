#!/bin/sh
python classify_dialect.py -e 500 -cb 0.9 -d beta09_1 -rd beta_reprod_09
python classify_dialect.py -e 500 -cb 0.9 -d beta09_2 -rd beta_reprod_09
python classify_dialect.py -e 500 -cb 0.9 -d beta09_3 -rd beta_reprod_09
python classify_dialect.py -e 500 -cb 0.9 -d beta09_4 -rd beta_reprod_09
python classify_dialect.py -e 500 -cb 0.9 -d beta09_5 -rd beta_reprod_09