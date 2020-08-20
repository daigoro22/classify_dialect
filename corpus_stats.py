import pandas as pd
import argparse
from word_and_categ_dict import WordAndCategDict

def words_count(line,wcdict):
    return len(wcdict.decode_as_pieces(line.tolist()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr','--train',type=str,default='corpus/train.pkl')
    parser.add_argument('-te','--test',type=str,default='corpus/test.pkl')
    args = parser.parse_args()

    wcdict = WordAndCategDict()

    df_train = pd.read_pickle(args.train)
    df_test = pd.read_pickle(args.test)

    print(len(df_train))
    print(len(df_test))

    df_train['WORDS_COUNT_D'] = df_train.apply(lambda x:words_count(x['DIALECT'],wcdict),axis=1)
    print(df_train['WORDS_COUNT_D'])