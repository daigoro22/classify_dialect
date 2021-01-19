import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from word_and_categ_dict import WordAndCategDict
import argparse
import fasttext
from chunk_dialect_classifier import CharacterOneHotEncoder

def apply_func_to_columns(data_frame,func,columns):
    for c in columns:
        data_frame[c] = data_frame.apply(lambda x:func(x[c]),axis=1)
    return data_frame

def train_test_valid_split(_df:pd.DataFrame,random_seed,train_ratio=0.6,test_and_valid_ratio=0.2):
    """データセットを train, test, valid に変換する関数.
    Args:
        _df (pd.DataFrame): データセット
        random_seed (float): ランダムに分割するときのシード値
        train_ratio (float): train データの比率. train_ratio + (test_and_valid_ratio * 2) < 1 になるように設定する.
        test_and_valid_ratio (float): test, valid のデータの比率. train_ratio + (test_and_valid_ratio * 2) < 1 になるように設定する.
    Returns:
        train (pd.DataFrame): train のデータセット
        test (pd.DataFrame): test のデータセット
        valid (pd.DataFrame): valid のデータセット
    >>> df=pd.DataFrame({'standard':['ア','イ','ウ','エ','オ','カ','キ','ク','ケ','コ'],'dialect':['ア','イ','ウ','エ','オ','カ','キ','ク','ケ','コ']})
    >>> tr, te, va = train_test_valid_split(df,random_seed=1,train_ratio=0.6,test_and_valid_ratio=0.2)
    >>> len(tr) / len(df)
    0.6
    >>> len(te) / len(df)
    0.2
    >>> len(va) / len(df)
    0.2
    """
    # 比率の合計は１でなくてはならない
    assert(train_ratio + (test_and_valid_ratio * 2) == 1.0)

    # valid の比率を計算
    test_ratio = test_and_valid_ratio
    valid_ratio = test_and_valid_ratio / (1 - test_and_valid_ratio)

    train,test  = train_test_split(_df,test_size=test_ratio,random_state=random_seed)
    train,valid = train_test_split(train,test_size=valid_ratio,random_state=random_seed)

    return train, test, valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--seed',type=int,default=1)
    parser.add_argument('-ft','--fasttext',type=str)
    parser.add_argument('-ac','--area_clf',action='store_true')
    parser.add_argument('-ch','--character',action='store_true')
    parser.add_argument('-chp','--character_pfts',nargs='*',default=[])
    parser.add_argument('-c','--corpus',type=str,default='corpus/concat_corpus_all.txt')
    parser.add_argument('-tr','--train_ratio',type=float,default=1/3)
    parser.add_argument('-tvr','--test_and_valid_ratio',type=float,default=1/3)
    args = parser.parse_args()

    wcdict = WordAndCategDict()

    if args.area_clf:
        index              = ['DIALECT','STANDARD','PFT','AREA']
        apply_atoi_columns = ['AREA']
        wcdict_area        = WordAndCategDict(categ_path='corpus/all_area.txt')
        data_postfix       = '_area'
    elif args.character:
        index              = ['DIALECT','STANDARD','PFT']
        apply_atoi_columns = []
        wcdict_area        = WordAndCategDict()
        wcdict             = WordAndCategDict(categ_list=args.character_pfts)
        data_postfix       = '_character' if args.character_pfts==[] else '_character_extracted'
        # カタカナのリスト
        classes  = [chr(i) for i in range(12449, 12532+1)]
        # 0-9 の数字のリスト
        classes += [str(n) for n in list(range(10))]
        # 伸ばし棒, 鼻濁音, 未知語, 人物名
        classes += ['ー','゜','*','X']
        encoder  = CharacterOneHotEncoder(classes=classes,sparse=False)
    else:
        index              = ['DIALECT','STANDARD','PFT']
        apply_atoi_columns = []
        wcdict_area        = WordAndCategDict()
        data_postfix       = ''

    apply_ctoi_columns = ['PFT']
    apply_func_columns = ['DIALECT','STANDARD']

    df = pd.read_csv(args.corpus,sep='\t',names=index)
    
    if args.fasttext:
        model         = fasttext.load_model(args.fasttext)
        func          = lambda x:[model[p] for p in wcdict.encode_as_pieces(x)]
        model_postfix = '_ft'
    # 抽出する県が指定されている場合, 抽出
    elif args.character and args.character_pfts != []:
        df = df[df['PFT'].isin(args.character_pfts)]
        func          = wcdict.stoi
        model_postfix = ''
    else:
        func          = wcdict.stoi
        model_postfix = ''

    # train, test, valid に分割
    train, test, valid = train_test_valid_split(
        _df=df,
        random_seed=args.seed,
        train_ratio=args.train_ratio,
        test_and_valid_ratio=args.test_and_valid_ratio)

    for tag,_df in zip(['train','test','valid'],[train,test,valid]):
        pkl_file = f'corpus/{tag}{model_postfix}{data_postfix}.pkl'

        if args.character:
            # 文字の one-hot ベクトルを取得して 県の列 と concat
            df_chara = encoder.get_encoded(_df,'DIALECT')
            df_chara = pd.concat([df_chara,encoder.get_encoded(_df,'STANDARD')],axis=1)
            df_chara = pd.concat([df_chara,_df['PFT']],axis=1)
            
            # 県名, 地域名->数値に変換
            df_chara = apply_func_to_columns(df_chara,wcdict.ctoi,apply_ctoi_columns)
            df_chara = apply_func_to_columns(df_chara,wcdict_area.ctoi,apply_atoi_columns)

            # 保存して後の処理をスキップして次へ
            df_chara.to_pickle(pkl_file)
            continue

        # 指定した関数を適用
        applied = apply_func_to_columns(_df,func,apply_func_columns)
        # 県名, 地域名->数値に変換
        applied = apply_func_to_columns(applied,wcdict.ctoi,apply_ctoi_columns)
        applied = apply_func_to_columns(applied,wcdict_area.ctoi,apply_atoi_columns)
        applied.to_pickle(pkl_file)