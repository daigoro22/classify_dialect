import sentencepiece as spm

def get_pieces_line(_file):
    for line in _file:
        pieces = [p.replace('▁','') for p in sp.EncodeAsPieces(line)]
        pieces = [p for p in pieces if p is not '']
        yield ' '.join(pieces)

if __name__ == "__main__":
    sp = spm.SentencePieceProcessor(model_file='spm/dialect_standard.model')
    
    with open('corpus/concat_dialect_standard.txt') as fr:
        with open('spm/concat_dialect_standard_spm.txt','w') as fw:
            for piece in get_pieces_line(fr):
                fw.write(piece + '\n')