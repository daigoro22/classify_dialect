import sentencepiece as spm

class SpmModel():
    def __init__(self,model_path='spm/dialect_standard.model'):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def decode_sentence(self,line):
        return self.sp.decode(line)
    
    def encode_as_pieces(self,line):
        pieces = [p.replace('▁','') for p in self.sp.EncodeAsPieces(line)]
        pieces = [p for p in pieces if p is not '']
        return pieces

if __name__ == "__main__":
    SpmModel()