#!/bin/sh
python classify_dialect.py -e 500 -cb 0.9 -d beta09 -rd beta
python classify_dialect.py -e 500 -cb 0.99 -d beta099 -rd beta
python classify_dialect.py -e 500 -cb 0.999 -d beta0999 -rd beta
python classify_dialect.py -e 500 -cb 0.9999 -d beta09999 -rd beta