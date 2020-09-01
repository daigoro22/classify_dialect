import chainer
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from dialect_and_area_classifier import DialectAndAreaClassifier
import matplotlib.pyplot as plt
from word_and_categ_dict import WordAndCategDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model',type=str,default='result/model.npz')
    parser.add_argument('-ts','--test_dataset',type=str,default='corpus/test_ft_area.pkl')
    args = parser.parse_args()
    
    model = DialectAndAreaClassifier(
        n_categ=48,
        n_embed=100,
        n_lstm=600,
        n_area=8
    )
    chainer.serializers.load_npz(args.model,model)

    df_test = pd.read_pickle(args.test_dataset)

    Y_categ = [p.tolist() for p in df_test['PFT'].values]
    Y_area = [a.tolist() for a in df_test['AREA'].values]

    X_dialect = [np.array(d) for d in df_test['DIALECT'].values]
    X_standard = [np.array(s) for s in df_test['STANDARD'].values]

    pred_categ, pred_area = model.predict(X_dialect,X_standard)

    cm_categ = confusion_matrix(Y_categ,pred_categ.tolist(),normalize='pred')
    cm_area = confusion_matrix(Y_area,pred_area.tolist(),normalize='pred')

    np.savez('result/cmat',pref=cm_categ,area=cm_area)

    wc_categ = WordAndCategDict()
    wc_area = WordAndCategDict(categ_path='corpus/all_area.txt')
    
    plt.figure(figsize=(10,10))
    cmd_categ = ConfusionMatrixDisplay(cm_categ,display_labels=wc_categ.categories()).plot(
        xticks_rotation='vertical'
        #ax=ax
    )
    plt.savefig('result/cmat_categ.png')

    cmd_area = ConfusionMatrixDisplay(cm_area,display_labels=wc_area.categories()).plot()
    plt.savefig('result/cmat_area.png')

