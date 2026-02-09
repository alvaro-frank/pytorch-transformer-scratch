import requests
import os

file_path = 'input.txt'
if not os.path.exists(file_path):
    print("A descarregar o dataset TinyShakespeare...")
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Tamanho do dataset: {len(text)} caracteres")

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocabulário único: {''.join(chars)}")
print(f"Tamanho do vocabulário: {vocab_size}")

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print("\n--- Teste de Sanidade ---")
frase_teste = "golden sun"
encoded = encode(frase_teste)
decoded = decode(encoded)

print(f"Original: {frase_teste}")
print(f"Encoded:  {encoded}")
print(f"Decoded:  {decoded}")

assert frase_teste == decoded, "Erro Crítico: O descodificador não recuperou o texto original!"
print("Teste Passou: Pipeline de dados sólido.")