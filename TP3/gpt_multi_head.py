import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 15000 #5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200
n_embd = 32
dropout = 0.2
n_heads = 4

if torch.cuda.is_available(): device = torch.device("cuda")
elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")

# ------------

torch.manual_seed(2023)

# Load the Victor Hugo dataset
with open('data/hugo_contemplations.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size, n_embd):
        super().__init__()
        # Create linear layers for key, query, and value
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Add dropout after softmax
        self.dropout = nn.Dropout(dropout)
        
        # Register the triangular matrix as a buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        
        # Apply key, query, and value projections
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)
        
        # Compute attention weights
        weights = q @ k.transpose(-2, -1) # (B, T, T)
        weights = weights / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32))
        
        # Apply causal mask
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax and dropout
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        out = weights @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        # Ensure that the head size makes sense with the embedding size
        assert n_embd % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        # Create a list of num_heads modules of type Head
        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(num_heads)])
        
        # Project the concatenated heads back to the original embedding dimension
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Add dropout after projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply each head to x and concatenate the results
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # Project the concatenated heads back to the original embedding dimension
        out = self.proj(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        return out


class FeedForward(nn.Module):
    """ a simple MLP with RELU and dropout """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ A single block of multi-head attention """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # Add two layer normalization layers
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        self.sa = MultiHeadAttention(n_head, head_size, n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        # Layer norm before multi-head attention, with skip connection
        x = x + self.sa(self.ln1(x))
        
        # Layer norm before feed-forward, with skip connection  
        x = x + self.ffwd(self.ln2(x))
        
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self, n_heads):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
    
        # Replace previous implementation with a Sequential of Blocks
        self.blocks = nn.Sequential(
            Block(n_embd, n_heads),
            Block(n_embd, n_heads),
            Block(n_embd, n_heads)
        )

        # Add layer normalization after the blocks
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        
        # Pass through the blocks of multi-head attention
        x = self.blocks(x)
        
        # Apply final layer normalization
        x = self.ln_final(x)
        
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(n_heads)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
prompt = torch.tensor(encode(['\n']), device=device)
context = torch.ones((1,1), dtype=torch.long, device=device) * prompt.to(device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))