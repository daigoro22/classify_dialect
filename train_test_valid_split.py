import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from word_and_categ_dict import WordAndCategDict
import argparse
import fasttext
import os

def apply_func_to_columns(data_frame,func,columns):
    for c in columns:
        data_frame[c] = data_frame.apply(lambda x:func(x[c]),axis=1)
    return data_frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--seed',type=int,default=1)
    parser.add_argument('-ft','--fasttext',type=str)
    parser.add_argument('-ac','--area_clf',action='store_true')
    parser.add_argument('-c','--corpus',type=str,default='corpus/concat_corpus_all.txt')
    args = parser.parse_args()

    if args.area_clf:
        index=['DIALECT','STANDARD','PFT','AREA']
        apply_atoi_columns = ['AREA']
        wcdict_area = WordAndCategDict(categ_path='corpus/all_area.txt')
        data_postfix = '_area'
    else:
        index = ['DIALECT','STANDARD','PFT']
        apply_atoi_columns = []
        wcdict_area = WordAndCategDict()
        data_postfix = ''

    apply_ctoi_columns = ['PFT']
    apply_func_columns = ['DIALECT','STANDARD']

    wcdict = WordAndCategDict()

    df = pd.read_csv(args.corpus,sep='\t',names=index)
    train,test = train_test_split(df,test_size=0.2,random_state=args.seed)
    train,valid = train_test_split(train,test_size=0.25,random_state=args.seed)
    
    if args.fasttext:
        model = fasttext.load_model(args.fasttext)
        func = lambda x:[model[p] for p in wcdict.encode_as_pieces(x)]
        model_postfix = '_ft'
    else:
        func = wcdict.stoi
        model_postfix = ''

    for tag,_df in zip(['_train','_test','_valid'],[train,test,valid]):
        applied = apply_func_to_columns(_df,func,apply_func_columns)
        applied = apply_func_to_columns(applied,wcdict.ctoi,apply_ctoi_columns)
        applied = apply_func_to_columns(applied,wcdict_area.ctoi,apply_atoi_columns)
        pkl_file = 'corpus/{}{}{}{}.pkl'.format(
            os.path.basename(args.corpus).split('.')[0],
            tag,
            model_postfix,
            data_postfix)
        applied.to_pickle(pkl_file)