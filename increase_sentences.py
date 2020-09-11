import numpy as np
import pandas as pd

if __name__ == "__main__":
    
    with open('corpus/concat_corpus.txt') as f:
        dialect = f.read().split('\n')

    with open('corpus/concat_corpus_standard_yomi.txt') as f:
        standard = f.read().split('\n')
    
    with open('corpus/all_pft.txt') as f:
        pft = f.read().split('\n')
    
    with open('corpus/all_area.txt') as f:
        area = f.read().split('\n')
    
    assert(len(pft) == len(area) == len(dialect) == len(standard))

    len_dlt = np.array([len(d.split('。')) for d in dialect])
    len_std = np.array([len(s.split('。')) for s in standard])

    len_diff = np.abs(len_dlt - len_std).tolist()

    new_dialect = [] 
    new_standard = []
    new_pft = []
    new_area = []

    for d,s,p,a,ld in zip(dialect,standard,pft,area,len_diff):
        if ld==0:
            d_splitted = d.split('。')
            s_splitted = s.split('。')
            if d_splitted[-1]=='':
                d_splitted = d_splitted[0:-1]
                s_splitted = s_splitted[0:-1]
            _len = len(d_splitted)
            new_dialect.extend(d_splitted)
            new_standard.extend(s_splitted)
            new_pft.extend([p]*_len)
            new_area.extend([a]*_len)

        else:
            new_dialect.append(d)
            new_standard.append(s)
            new_pft.append(p)
            new_area.append(a)
    
    assert(
        len(new_pft) == 
        len(new_area) == 
        len(new_dialect) == 
        len(new_standard))
    
    df = pd.DataFrame({
        'dialect': new_dialect,
        'standard': new_standard,
        'pft': new_pft,
        'area': new_area
    })
    counts = df.groupby('pft').value_counts()
    df.to_csv('corpus/concat_corpus_inc.txt',sep='\t',index=False,header=False3)