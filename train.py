import torch
from src.model import BigramLanguageModel
from src.data_loader import CharTokenizer

batch_size = 32
block_size = 8
max_iters = 3000
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300

tokenizer = CharTokenizer(file_path='data/input.txt', data_url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
tokenizer.load()

data = torch.tensor(tokenizer.encode(tokenizer.data), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Vocabulary: {tokenizer.vocab_size} chars")
print(f"Train data: {len(train_data)} tokens")

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y

model = BigramLanguageModel(tokenizer.vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Model has {sum(p.numel() for p in m.parameters())} parameters")
print("Starting training...\n")

for iter in range(max_iters):
    if iter % eval_interval == 0:
        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)
        print(f"Step {iter}: Loss {loss.item():.4f}")

    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()                       
    optimizer.step()                      

print(f"\n--- Treino Concluído ---")
print(f"Loss Final: {loss.item():.4f}")

print("\nGerando texto (com modelo Bigram)...")
idx = torch.zeros((1, 1), dtype=torch.long, device=device)