import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from word_and_categ_dict import WordAndCategDict
import argparse
import fasttext
from chunk_dialect_classifier import CharacterLabelEncoder

def apply_func_to_columns(data_frame,func,columns):
    for c in columns:
        data_frame[c] = data_frame.apply(lambda x:func(x[c]),axis=1)
    return data_frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--seed',type=int,default=1)
    parser.add_argument('-ft','--fasttext',type=str)
    parser.add_argument('-ac','--area_clf',action='store_true')
    parser.add_argument('-ch','--character',action='store_true')
    parser.add_argument('-chp','--character_pfts',nargs='*',default=[])
    parser.add_argument('-c','--corpus',type=str,default='corpus/concat_corpus_all.txt')
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
        encoder  = CharacterLabelEncoder(classes=classes)
    else:
        index              = ['DIALECT','STANDARD','PFT']
        apply_atoi_columns = []
        wcdict_area        = WordAndCategDict()
        data_postfix       = ''

    apply_ctoi_columns = ['PFT']
    apply_func_columns = ['DIALECT','STANDARD']

    df          = pd.read_csv(args.corpus,sep='\t',names=index)
    train,test  = train_test_split(df,test_size=0.2,random_state=args.seed)
    train,valid = train_test_split(train,test_size=0.25,random_state=args.seed)
    
    if args.fasttext:
        model         = fasttext.load_model(args.fasttext)
        func          = lambda x:[model[p] for p in wcdict.encode_as_pieces(x)]
        model_postfix = '_ft'
    else:
        func          = wcdict.stoi
        model_postfix = ''

    for tag,_df in zip(['train','test','valid'],[train,test,valid]):
        pkl_file = f'corpus/{tag}{model_postfix}{data_postfix}.pkl'

        if args.character:
            # 文字の one-hot ベクトルを取得して 県の列 と concat
            df_chara = encoder.get_encoded(_df,'DIALECT')
            df_chara = pd.concat([df_chara,encoder.get_encoded(_df,'STANDARD')],axis=1)
            df_chara = pd.concat([df_chara,_df['PFT']],axis=1)

            # 抽出する県が指定されている場合, 抽出
            if args.character_pfts != []:
                df_chara = df_chara[df_chara['PFT'].isin(args.character_pfts)]
            
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