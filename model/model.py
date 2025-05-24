from layers import *


class Decoder(Module):

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
        
        self.c_mha = CausalAttention(dim, heads, max_tokens, bias, layer_idx=layer_idx, device=device)
        self.ffn = FeedForward(dim, expansion_scale, bias, gate, nl, device, layer_idx)
        
        self.l1 = nn.LayerNorm(dim)
        self.l2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor, mask: Optional[Tensor], shift: int = 0) -> Tensor: 
        x = self.c_mha(self.l1(x), None, mask, shift) + x 
        x = self.ffn(self.l2(x), None) + x
        return x

class GPT(Module): 

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
                 device: str = "cuda",
                 ):
        super().__init__()
    
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim, device=device)

        self.blocks = []
    
        for i in range(num_layers):
            self.blocks.append(Decoder(dim, heads, max_tokens, gate, bias, expansion_scale, nl, device, i))

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
        x = self.embed(x)
        
        mask = self._gen_mask(x, shift)
       
        for block in self.blocks:
            x = block(x, mask, shift)

        return F.softmax(self.cls(x) / temp, dim = -1)
