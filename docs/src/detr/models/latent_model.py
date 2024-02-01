import torch.nn as nn
from torch.nn import functional as F
import torch

DROPOUT_RATE = 0.1

# a causal transformer block
class Causal_Transformer_Block(nn.Module):
    def __init__(self, seq_len, latent_dim, num_head) -> None:
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.ln_1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_head, dropout=DROPOUT_RATE, batch_first=True)
        self.ln_2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(DROPOUT_RATE),
        )

        # self.register_buffer("attn_mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())
    
    def forward(self, x):
        attn_mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool), diagonal=1)
        x = self.ln_1(x)
        x = x + self.attn(x, x, x, attn_mask=attn_mask)[0]
        x = self.ln_2(x)
        x = x + self.mlp(x)
        
        return x

# use self-attention instead of RNN to model the latent space sequence
class Latent_Model_Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, latent_dim=256, num_head=8, num_layer=3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.input_layer = nn.Linear(input_dim, latent_dim)
        self.weight_pos_embed = nn.Embedding(seq_len, latent_dim)
        self.attention_blocks = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            *[Causal_Transformer_Block(seq_len, latent_dim, num_head) for _ in range(num_layer)],
            nn.LayerNorm(latent_dim)
        )
        self.output_layer = nn.Linear(latent_dim, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = x + self.weight_pos_embed(torch.arange(x.shape[1], device=x.device))
        x = self.attention_blocks(x)
        logits = self.output_layer(x)

        return logits
    
    @torch.no_grad()
    def generate(self, n, temperature=0.1, x=None):
        if x is None:
            x = torch.zeros((n, 1, self.input_dim), device=self.weight_pos_embed.weight.device)
        for i in range(self.seq_len):
            logits = self.forward(x)[:, -1]
            probs = torch.softmax(logits / temperature, dim=-1)
            samples = torch.multinomial(probs, num_samples=1)[..., 0]
            samples_one_hot = F.one_hot(samples.long(), num_classes=self.output_dim).float()
            x = torch.cat([x, samples_one_hot[:, None, :]], dim=1)
        
        return x[:, 1:, :]

