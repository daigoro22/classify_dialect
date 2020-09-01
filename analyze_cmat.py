import numpy as np
from word_and_categ_dict import WordAndCategDict
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    cmat = np.load('result/cmat.npz')['pref']
    wcdict_pref = WordAndCategDict()
    mask_array = np.full_like(cmat,False,dtype=np.bool)
    for i,c in enumerate(cmat):
        true_i = wcdict_pref.itoc(i)
        val = [(i,c[i])]
        _val = [(j,v) for j,v in enumerate(c) if j!=i]
        _val = sorted(_val, key=lambda x:x[1],reverse=True)[0:2]
        val.extend(_val)
        val = [x[0] for x in val]
        mask_array[i][val] = True
    
    new_cmat = cmat * mask_array
    
    plt.figure(figsize=(10,10))
    cmd_categ = ConfusionMatrixDisplay(new_cmat,display_labels=wcdict_pref.categories()).plot(
        xticks_rotation='vertical',
        include_values=False
        #ax=ax
    )
    cmat_df = pd.DataFrame(new_cmat,columns=wcdict_pref.categories(),index=wcdict_pref.categories())
    cmat_df.to_csv('result/cmat_categ.csv')
    #np.savetxt('result/cmat_categ.csv',new_cmat,delimiter=',',fmt='%.2f')
    plt.savefig('result/cmat_categ_new.png')
    '''
    fig,ax = plt.subplots(
        figsize=((len(wcdict_pref.categories())+1)*1.2,
        (len(new_cmat)+1)*0.4))
    ax.axis('off')
    tbl = ax.table(cellText=new_cmat,
        bbox=[0,0,1,1],
        colLabels=wcdict_pref.categories(),
        rowLabels=wcdict_pref.categories())
    plt.savefig('result/cmat_categ_new.png')
    '''