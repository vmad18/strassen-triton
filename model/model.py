from .layers import *


class Decoder(nn.Module):

    def __init__(self, 
                 dim: int, 
                 heads: int, 
                 max_tokens: int, 
                 gate: bool = False,
                 bias: bool = True,
                 expansion_scale: int = 4,
                 nl = F.silu,
                 device: str = "cuda",
                 layer_idx: Optional[int] = None
                 ):
        super().__init__()
        
        self.c_mha =         CausalAttention(dim, heads, max_tokens, bias, layer_idx=layer_idx, device=device)
# nn.MultiheadAttention(dim, heads, bias=False, batch_first=True, device=device) 
        self.ffn = FeedForward(dim, expansion_scale, bias, gate, nl, device, layer_idx)
        
        self.l1 = nn.LayerNorm(dim, device=device)
        self.l2 = nn.LayerNorm(dim, device=device)

    def forward(self, x: Tensor, mask: Optional[Tensor], shift: int = 0, padding_mask = None) -> Tensor:
        seq_len = x.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        pre_norm = self.l1(x)
        x = self.c_mha(pre_norm, None, causal_mask, shift) + x 
        x = self.ffn(self.l2(x)) + x
        return x



class GPT(nn.Module): 

    def __init__(self,
                 dim: int, 
                 heads: int, 
                 max_tokens: int, 
                 gate: bool = False,
                 bias: bool = True,
                 expansion_scale: int = 4,
                 nl = F.silu,
                 num_layers: int = 8,
                 vocab_size: int = 65536,
                 PAD_TOKEN_ID = -1, 
                 device: str = "cuda",
                 ):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim, device=device)
        # self.pos_embed = PositionalEncoding(dim, max_tokens, device)
        
        self.blocks = nn.ModuleList([])
    
        for i in range(num_layers):
            self.blocks.append(Decoder(dim, heads, max_tokens, gate, bias, expansion_scale, nl, device, i))

        self.PAD_TOKEN_ID = PAD_TOKEN_ID
        self.cls = nn.Linear(dim, vocab_size, device=device)

    def _gen_mask(self,
                  input: Tensor,
                  shift: int = 0) -> Tensor:
        curr_seq = input.shape[-1]

        target = curr_seq + shift
        shift = shift + 1

        return causal_mask(curr_seq,
                           target,
                           shift=shift)[None, None, ...]

    def forward(self, x: Tensor, 
                shift: int = 0, 
                temp: float = 1.) -> Tensor:
        padding_mask = (x == self.PAD_TOKEN_ID)
        mask = self._gen_mask(x, shift)
        x = self.embed(x)
        # x = self.pos_embed(x)
        
        for block in self.blocks:
            x = block(x, mask, shift, padding_mask)

        return self.cls(x) # F.softmax(self.cls(x) / temp, dim = -1)
