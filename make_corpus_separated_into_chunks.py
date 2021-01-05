import glob
import re 
import MeCab

def assert_lines(list_std,list_dia):
    """元コーパスの全行の Assertion をまとめて行う関数.
    方言と共通語でコーパスの行数と, 会話番号に対する行番号が揃っているかを Assert する.
    Args:
        list_std (list of str): コーパスの共通語の全行
        list_dia (list of str): コーパスの方言の全行        
    """
    # 会話番号の正規表現
    PATT_PRE_DLG = '[0-9]+(?:[Ａ-Ｚ]|[A-Z])(?:：|:)'

    # 方言と共通語で行数が揃っているかどうかの Assertion
    assert(len(list_dia)==len(list_std))

    # 方言と共通語で会話番号の行が揃っているかの Assertion
    # 会話番号を抽出する関数
    get_dlg = lambda line: re.findall(PATT_PRE_DLG,line)[0]
    
    # 会話番号:行番号の辞書を作成
    dict_dlg_s = {get_dlg(s):i for i,s in enumerate(list_std) if re.match(PATT_PRE_DLG,s)}
    dict_dlg_d = {get_dlg(d):i for i,d in enumerate(list_dia) if re.match(PATT_PRE_DLG,d)}
    
    # 方言と共通語で, 各会話番号に対応する行番号が揃っているかどうか assert
    for k,v in dict_dlg_s.items():
        # 会話番号の半角, 全角が間違っている場合はそのまま continue
        if k not in dict_dlg_d:
            print(f'key {k} not in {path_d}')
            continue
        assert(dict_dlg_d[k] == v)

def shape_lines(lines):
    """コーパスの行を整理する関数.
    以下の表現を削除する.
    * { }: 笑，咳，咳払い，間，などの非言語音。
    * ×××: 言い間違いや言い淀みなど。
    * [＝ ]: 意味の説明や，意訳であることを示す.
    * ＝[　]:意訳の表記間違い.大阪にのみ存在する.
    * | |: 注意書きなど.
    *〔 〕:注記。方言形の意味・用法，特徴的音声などについて説明し，文字化・ 共通語訳の後にまとめてある。
    * ――　中　略　――: 中略を表す.
    以下の表現は置換する.
    * [ ]: 方言音声には出てこないが，共通語訳の際に補った部分. カッコの中身で置換.
    * (): あいづち. ひとりの人が連続して話している時にさえぎったり，口をはさんだりした個所. カッコの中身で置換.
    * Xn: nは数字. 会話に参加していない他者を示す. 『X』で置換.
    * ｎ: 高知県にのみ含まれる. 非言語音. 『ン』で置換.
    * \*\*\*: 聞き取れない部分. 未知語『<unk>』で置換.
    * ///: 対応する共通語訳が不明な部分. 未知語『<unk>』で置換.
    Args:
        line (str): 元コーパスの全行.
    Returns:
        lines_shaped (str): 整形されたコーパスの全行.
    """

    # 会話番号の正規表現
    PATT_PRE_DLG = '。?\n?[0-9]+(?:[Ａ-Ｚ]|[A-Z])(?:：|:)'
    # 削除する部分の正規表現
    PATT_STRIP   = r'(?:｛.*?｝|｜.*?｜|［(?:＝|=).*?］|×+|x+|〔.*?〕|――　中　略　――|（(?:[Ａ-Ｚ]|[A-Z]).*?）|［|］|＝［.*?］)'
    # 会話に参加していない人の名前の正規表現
    PATT_NAME    = r'((?:Ｘ|X)(?:[0-9]|[０-９])+)|[Ａ-Ｚ]'
    # 聞き取れなかった部分の正規表現
    PATT_UNREC   = r'＊+|\*+|／+|/+'

    # 非言語音『n』を置換
    lines_shaped = lines.replace('ｎ','ン')
    # 非言語音, 言い間違い, 説明, 注意書き, 注記, 中略を削除
    lines_shaped = re.sub(PATT_STRIP,'',lines_shaped)
    # 会話に参加していない人物の名前をXに置換
    lines_shaped = re.sub(PATT_NAME,'X',lines_shaped)
    # 聞き取れなかった部分は 未知語に置換
    lines_shaped = re.sub(PATT_UNREC,'<unk>',lines_shaped)
    # 会話番号を句点と改行に置換
    lines_shaped = re.sub(PATT_PRE_DLG,'。\n',lines_shaped)

    return lines_shaped.split('\n')

if __name__ == "__main__":
    tagger = MeCab.Tagger('-Oyomi -d /var/lib/mecab/dic/ipadic-neologd')
    
    path_std_list = glob.glob('corpus/furusato/*/*/*009.txt')
    path_dia_list = glob.glob('corpus/furusato/*/*/*008.txt')

    PATT_PREF    = 'corpus/furusato/.+/(.+)/.+\.txt'

    list_new_lines = []
    for path_s,path_d in zip(path_std_list,path_dia_list):
        
        # フォルダ名から県の名前取得
        pref = re.findall(PATT_PREF,path_s)[0]

        with open(path_s,encoding='shift-jis') as fs:
            list_std = fs.read().replace('\n\n','\n').split('\n')
        with open(path_d,encoding='shift-jis') as fd:
            list_dia = fd.read().replace('\n\n','\n').split('\n')

        # 群馬は元コーパスの共通語が575番目の行で揃っていないので削除
        if pref == 'gunma':
            del list_dia[575]
        
        # 方言と共通語の Assertion
        assert_lines(list_std,list_dia)

        # 全ての行を整形する
        list_shaped_std = shape_lines('\n'.join(list_std))
        list_shaped_dia = shape_lines('\n'.join(list_dia))
        
        list_new_line = []

        sep = lambda x:x.split('　')
        PATT_DELETE_CHUNK = '。'
        is_necessary = lambda x:(x!='' and not re.fullmatch(PATT_DELETE_CHUNK,x))
        
        list_sep_std = []
        list_sep_dia = []
        for std, dia in zip(list_shaped_std,list_shaped_dia):
            list_std = [s for s in sep(std) if is_necessary(s)]
            list_dia = [d for d in sep(dia) if is_necessary(d)]
            if list_std == [] or list_dia == []:
                continue
            if len(list_std) == len(list_dia):
                list_sep_std.extend(list_std)
                list_sep_dia.extend(list_dia)

        assert(len(list_sep_std) == len(list_sep_dia))

        parse = lambda x: tagger.parse(x).strip().replace('。','')
        list_new_line = ['\t'.join((parse(d),parse(s),pref)) + '\n' for s,d in zip(list_sep_std,list_sep_dia)]
        # 行全てをリストに追加
        list_new_lines.extend(list_new_line)
    
    with open('corpus/corpus_separated_into_chunks.tsv','w') as f:
        f.writelines(list_new_lines)
