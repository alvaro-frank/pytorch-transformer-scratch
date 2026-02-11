import requests
import os

class CharTokenizer:
    def __init__(self, file_path, data_url=None):
        self.file_path = file_path
        self.data_url = data_url
        self.stoi = None
        self.itos = None
        self.vocab_size = 0
        self.data = None

    def load(self):
        if not os.path.exists(self.file_path):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            response = requests.get(self.data_url)
            response.raise_for_status()
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()
            
        self._build_vocab()

    def _build_vocab(self):
        chars = sorted(list(set(self.data)))
        self.vocab_size = len(chars)
        
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, indices):
        return ''.join([self.itos[i] for i in indices])

if __name__ == "__main__":
    tokenizer = CharTokenizer(file_path='data/input.txt', data_url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
    tokenizer.load()
    
    print(f"Vocabulary: {tokenizer.vocab_size} unique chars")

    encoded = tokenizer.encode("golden sun")
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: golden sun")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")