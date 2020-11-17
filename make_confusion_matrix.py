import chainer
import argparse
import pandas as pd
import cupy as np
#import cupy as cp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from dialect_and_area_classifier import DialectAndAreaClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from word_and_categ_dict import WordAndCategDict
from cb_loss_classifier import CbLossClassifier
import os

def get_confusion_matrix_DAC(df_test,model,normalize='all'):
    Y_categ    = [p.tolist() for p in df_test['PFT'].values]
    Y_area     = [a.tolist() for a in df_test['AREA'].values]
    X_dialect  = [np.array(d,dtype=np.float32) for d in df_test['DIALECT'].values]
    X_standard = [np.array(s,dtype=np.float32) for s in df_test['STANDARD'].values]
    
    pred_categ, pred_area = model.predict(X_dialect,X_standard)
    
    cm_categ = confusion_matrix(Y_categ,pred_categ.tolist(),normalize=normalize)
    cm_area  = confusion_matrix(Y_area,pred_area.tolist(),normalize=normalize)
    return cm_categ, cm_area

def save_cmat_fig(cmat,ticklabels,filename):
    df = pd.DataFrame(cmat, columns=ticklabels,index=ticklabels)
    
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111)
    sns.heatmap(
        df,
        ax          = ax,
        xticklabels = ticklabels,
        yticklabels = ticklabels,
        linewidth   = 0.1)
    fig.savefig(filename)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model',type=str,default='result/model.npz')
    parser.add_argument('-ts','--test_dataset',type=str,default='corpus/test_ft_area.pkl')
    parser.add_argument('-mt','--model_type',choices=['DAC','CB'],default='CB')
    args = parser.parse_args()
    
    if args.model_type == 'DAC':
        model = DialectAndAreaClassifier(
            n_categ=48,
            n_embed=100,
            n_lstm=600,
            n_area=8
        )
    elif args.model_type == 'CB':
        model = CbLossClassifier(
            spc_list=[],
            beta=0.9,
            n_categ=48,
            n_embed=100,
            n_lstm=600,
            n_area=8
        )
    chainer.serializers.load_npz(args.model,model)

    df_test = pd.read_pickle(args.test_dataset)
    cm_categ,cm_area = get_confusion_matrix_DAC(df_test,model)
    # np.savez('result/cmat',pref=cm_categ,area=cm_area)
    wd = WordAndCategDict()
    save_cmat_fig(
        cm_categ,
        wd.categories(),
        'cmat_'+os.path.basename(args.model).split('.')[0])

