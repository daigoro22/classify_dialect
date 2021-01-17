import pandas as pd
import argparse
import matplotlib.pyplot as plt
from icecream import ic

if __name__ == "__main__":
    """
    コーパスの統計情報（要素のカウント, 単語の長さ）を表示する.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--corpus',default='corpus/corpus_separated_into_chunks.tsv')
    parser.add_argument('-cl','--columns',nargs='*',default=['dialect','standard','pref'])
    parser.add_argument('-ck','--count_key',default='pref')
    parser.add_argument('-wck','--word_count_key',nargs='*',default=['dialect','standard'])
    parser.add_argument('-rd','--result_directory',default='result/')
    parser.add_argument('-d','--description',default='chunks')
    args = parser.parse_args()

    # コーパス読み込み
    df = pd.read_table(args.corpus)
    df.columns = args.columns

    # 事例数出力
    ic(len(df))

    fig = plt.figure()

    # count_key で指定したカラムの要素ごとのカウントをbarプロットする
    se_value_counts = df[args.count_key].value_counts()
    se_value_counts.plot(kind='bar')
    plt.savefig(f'{args.result_directory}value_counts_{args.description}.png')

    # 最小値と最大値出力
    ic(se_value_counts.max,se_value_counts.min(),se_value_counts.mean(),se_value_counts.var())

    # word_count_key で指定したカラムの要素の文字数を新しい列に追記してbarプロットする
    # 文字数を格納する新しい列の名前
    list_applied_wck = [f'word_counts_{wck}' for wck in args.word_count_key]
    
    # 軸目盛りのためのユニークな文字数のリスト
    list_unique_nums = []
    # 新しい列に文字数を格納し, ユニークな文字数で list_unique_nums を更新していく
    for a_wck, wck in zip(list_applied_wck,args.word_count_key):
        df[a_wck] = df[wck].apply(lambda x:len(x))
        list_unique_nums.extend(df[a_wck].unique().tolist())

        # 文字数の value_counts
        se_vc_wck = df[a_wck].value_counts()
        # 最大値, 最小値, モード, 平均, 分散を出力
        ic(a_wck,se_vc_wck.max(),se_vc_wck.min(),df[a_wck].mode(),se_vc_wck.var(),se_vc_wck.mean())

    # 再度ユニークな文字数のリストを取得してソート
    list_unique_nums = sorted(list(set(list_unique_nums)))
    # 軸ラベル
    list_label_nums  = [str(n) for n in list_unique_nums]
    # ヒストグラムでプロットする
    df[list_applied_wck].plot(kind='hist',bins=list_unique_nums,alpha=0.5)
    # 軸目盛りの設定 plot() を実行した後でないと反映されないので注意
    plt.xticks(list_unique_nums,list_label_nums,rotation=90)
    plt.savefig(f'{args.result_directory}word_counts_{args.description}.png')
