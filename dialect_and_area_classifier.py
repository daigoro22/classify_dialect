import chainer
import chainer.links as L
import chainer.functions as F
import cupy as np

class DialectAndAreaClassifier(chainer.Chain):
    def __init__(self,n_categ,n_area,n_embed,n_lstm):
        super(DialectAndAreaClassifier,self).__init__()
        with self.init_scope():
            self.lstm = L.NStepLSTM(1,n_embed,n_lstm,dropout=0.2)
            self.categ = L.Linear(None,n_categ)
            self.area = L.Linear(None,n_area)

    def __call__(self,dialect,standard,category,area):
        h_emb_d = dialect 
        h2,_,_ = self.lstm(None,None,h_emb_d)
        h3 = F.relu(h2[0])
        pred_categ = self.categ(h3)
        pred_area = self.area(h3)

        loss_categ = F.softmax_cross_entropy(pred_categ,category)
        loss_area = F.softmax_cross_entropy(pred_area,area)

        acc_categ = F.accuracy(pred_categ,category)
        acc_area = F.accuracy(pred_area,area)

        loss = (loss_categ + loss_area) / 2
        accuracy = (acc_categ + acc_area) / 2
        chainer.reporter.report({'loss':loss},self)
        chainer.reporter.report({'accuracy':accuracy},self)
        chainer.reporter.report({'acc_categ':acc_categ},self)
        chainer.reporter.report({'acc_area':acc_area},self)
        return loss
    
    def predict(self,dialect,standard):
        h_emb_d = dialect 
        h2,_,_ = self.lstm(None,None,h_emb_d)
        h3 = F.relu(h2[0])
        pred_categ = np.argmax(np.array(self.categ(h3).data),axis=1,dtype=np.int32)
        pred_area = np.argmax(np.array(self.area(h3).data),axis=1,dtype=np.int32)
        
        return pred_categ,pred_area
    
    def pred_area(self,dialect,standard):
        _,pred_area = self.predict(dialect,standard)
        return pred_area
    
    def pred_categ(self,dialect,standard):
        pred_categ,_ = self.predict(dialect,standard)
        return pred_categ

def batch_converter_area(batch,device):
    dialect = [np.array(b[0]) for b in batch]
    standard = [np.array(b[1]) for b in batch]
    category = np.array([b[2] for b in batch],dtype=np.int32)
    area = np.array([b[3] for b in batch],dtype=np.int32)
    return {'dialect':dialect,'standard':standard,'category':category,'area':area}