import os
from pathlib import Path

#Tokenizer from scratch on vocabulary of corpus
from tokenizers import ByteLevelBPETokenizer

def train_tokenizer():
    tokenizer = ByteLevelBPETokenizer(lowercase=True)
    paths = [str(x) for x in Path(".").glob("text_split/*.txt")]

    tokenizer.train(
        files=paths, 
        vocab_size=10000, 
        min_frequency=2, 
        show_progress=True,
        special_tokens=["<s>","<pad>","<e>","<unk>","<mask>",]
    )

    os.mkdir('Byte_tokenizer')
    tokenizer.save_model('Byte_tokenizer')

if __name__ == '__main__':
    train_tokenizer()