from collections import defaultdict
import sentencepiece as spm
import numpy as np

class WordAndCategDict():
    
    def __init__(self,model_path='spm/dialect_standard.model',categ_path='corpus/all_pft.txt',categ_list=[]):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size = self.sp.get_piece_size()
        if categ_list==[]:
            with open(categ_path,'r') as f:
                categ_list = [c.strip() for c in f]
        self.__itoc = sorted(set(categ_list),key=categ_list.index)
        self.__ctoi = {c:i for i,c in enumerate(self.__itoc)}

    def itos(self,line):
        return self.sp.decode(line)

    def stoi(self,line):
        if line is not np.nan:
            return np.array(self.sp.encode(line),dtype=np.int32)
        else:
            return np.zeros((1,),dtype=np.int32)
    
    def ctoi(self,categ):
        return np.array(self.__ctoi[categ],dtype=np.int32)
    
    def itoc(self,index):
        return self.__itoc[index]
    
    def categories(self):
        return self.__itoc
    
    def decode_sentence(self,line):
        return self.sp.decode(line)
    
    def encode_as_pieces(self,line):
        if line is not np.nan:
            pieces = [p.replace('▁','') for p in self.sp.EncodeAsPieces(line)]
            pieces = [p for p in pieces if p is not '']
        else:
            pieces = ['<unk>']
        return pieces
    
    def decode_as_pieces(self,line):
        if line is not np.nan:
            pieces = [p.replace('▁','') for p in self.sp.id_to_piece(line)]
            pieces = [p for p in pieces if p is not '']
        else:
            pieces = ['<unk>']
        return pieces

if __name__ == "__main__":
    wcd1 = WordAndCategDict()
    wcd2 = WordAndCategDict()
    print(wcd2.categories())