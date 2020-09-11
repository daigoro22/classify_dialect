import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as path
from word_and_categ_dict import WordAndCategDict
from train_test_valid_split import apply_func_to_columns
import fasttext

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--train',default='corpus/concat_corpus_inc.txt') # train データセット
    parser.add_argument('-s','--size',default=600) # 増幅するサイズ
    parser.add_argument('-k','--key',default='PFT') # 増幅する際のキー
    parser.add_argument('-ft','--fasttext',default='fasttext_model/model')
    args = parser.parse_args()

    # train データセット読み込み
    df_train = pd.read_csv(args.train,sep='\t',names=['DIALECT','STANDARD','PFT','AREA'])

    # key ごとの分布を取得
    se_counts = df_train[args.key].value_counts()

    # over-sampling する key を抽出
    se_os = se_counts.where(se_counts<args.size).dropna()
    se_os = se_os.apply(lambda x:round(args.size/x))

    # df のある列を増幅する関数
    def duplicate(df,column,num):
        return df[column].tolist() * num

    # 結果が出力されるdf
    df_os = pd.DataFrame()

    # over-sampling しない key を抽出
    set_all_pft = set(df_train[args.key].value_counts().index.tolist())
    set_not_os_pft = set_all_pft - set(se_os.index.tolist())

    # over-sampling しない key の要素は増幅せずに結合
    for i in set_not_os_pft:
        df_i = df_train[df_train[args.key]==i]
        df_os = pd.concat([df_os,df_i])

    # over-sampling する key の要素は増幅して結合
    for i,s in se_os.iteritems():
        df_i = df_train[df_train[args.key]==i] # key が一致する要素を抽出
        df_duplicated = pd.DataFrame(
            {c:duplicate(df_i,c,s) for c in df_i.columns}) # 同じカラムで増幅したdf
        df_os = pd.concat([df_os,df_duplicated]) # df_os に増幅した df を結合する
    
    # 結果の分布を出力
    se_os_counts = df_os[args.key].value_counts()
    plt.xticks(rotation=90)
    sns.barplot(se_os_counts.index,se_os_counts.values)
    plt.savefig('result/dist_sentences_os.png')


    # 文, 県, エリアの分散表現への変換
    model = fasttext.load_model(args.fasttext)
    wcdict = WordAndCategDict()
    wcdict_area = WordAndCategDict(categ_path='corpus/all_area.txt')
    func = lambda x:[model[p] for p in wcdict.encode_as_pieces(x)]

    # 分散表現化する関数を各列に適用
    applied = apply_func_to_columns(df_os,func,['DIALECT','STANDARD'])
    applied = apply_func_to_columns(applied,wcdict.ctoi,['PFT'])
    applied = apply_func_to_columns(applied,wcdict_area.ctoi,['AREA'])

    # df_os を保存
    pkl_path = 'corpus/{}_train_ft_area_os.pkl'.format(path.basename(args.train).split('.')[0])
    applied.to_pickle(pkl_path)

