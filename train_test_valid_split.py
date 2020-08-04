import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from word_and_categ_dict import WordAndCategDict
import argparse
import fasttext

def to_index_and_save(data_d,data_s,data_c,wcdict,data_name,index,func=None):
    data_name += '.pkl'
    func = wcdict.stoi if func == None else func

    data_d_new = data_d.map(func)
    data_s_new = data_s.map(func)
    data_c_new = data_c.map(wcdict.ctoi)

    pd.concat([data_d_new,data_s_new,data_c_new],axis=1).to_pickle(data_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--seed',type=int,default=1)
    parser.add_argument('-ft','--fasttext',type=str)
    args = parser.parse_args()

    index=['DIALECT','STANDARD','PFT']
    df = pd.read_csv('corpus/concat_corpus_all.txt',sep='\t',names=index)
    train,test = train_test_split(df,test_size=0.2,random_state=args.seed)
    train,valid = train_test_split(train,test_size=0.25,random_state=args.seed)
    
    wcdict = WordAndCategDict()

    if args.fasttext:
        model = fasttext.load_model(args.fasttext)
        func = lambda x:[model[p] for p in wcdict.encode_as_pieces(x)]
        model_postfix = '_ft'
    else:
        func = None
        model_postfix = ''

    to_index_and_save(
        train['DIALECT'],
        train['STANDARD'],
        train["PFT"],
        wcdict,
        'corpus/train'+model_postfix,
        index,
        func)

    to_index_and_save(
        test['DIALECT'],
        test['STANDARD'],
        test["PFT"],
        wcdict,
        'corpus/test'+model_postfix,
        index,
        func)

    to_index_and_save(
        valid['DIALECT'],
        valid['STANDARD'],
        valid["PFT"],
        wcdict,
        'corpus/valid'+model_postfix,
        index,
        func)