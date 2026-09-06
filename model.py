import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(self.vocab_size,self.d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):

    def __init__(self,d_model,seq_len, dropout):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        pe=torch.zeros((seq_len,d_model))

        pos = torch.arange(0, seq_len).unsqueeze(1)
        i=torch.arange(0,self.d_model//2)
        denom=torch.pow(10000,(2*i)/d_model)

        pe[:,0::2]=torch.sin(pos/denom)
        pe[:,1::2]=torch.cos(pos/denom)

        pe=pe.unsqueeze(0)# shape->(1,seq_len,d_model)

        self.register_buffer("pe",pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps=10**-6):
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))
        self.bias=nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean=x.mean(dim=-1,keep_dim=True)
        std=x.std(dim=-1,keep_dim=True)

        return self.alpha*(x-mean)/(std+self.eps) +self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, d_ff,dropout):
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model,h,dropout):
        super().__init__()
        self.d_model=d_model
        self.h=h

        assert d_model%h==0, "d_model is not divisible by number of heads"

        self.d_k=d_model//h

        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)

        self.dropout=nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k=query.shape[-1]
        attention_score=(query @ key.transpose(-2,-1))/math.sqrt(d_k)

        if mask:
            attention_score.masked_fill(mask==0,-1e9)

        attention_score=attention_score.softmax(dim=-1)

        if dropout:
            attention_score=dropout(attention_score)
        return (attention_score@value),attention_score

    def forward(self, q, k, v, mask):
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)

        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,attention_score=MultiHeadAttention.attention(query,key,value,mask,self.dropout)
        x=x.transpose(1,2).contiquous().view(x.shape[0],-1,self.h*self.d_k)

        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self,dropout):

        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self,self_attention_block:MultiHeadAttention,feed_forward_block:FeedForwardBlock,dropout:float):
        super().__init__()
        self.attention_block=self_attention_block
        self.feed_forward=feed_forward_block
        self.residual_connection=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x=self.residual_connection[0](x,lambda x:self.attention_block(x,x,x,src_mask))
        x=self.residual_connection[1](x,self.feed_forward)
        return x

class Encoder(nn.Module):

    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block:MultiHeadAttention, cross_attention_block:MultiHeadAttention,feed_forward_block:FeedForwardBlock,dropout:float):

        super().__init__()
        self.self_attention_block=self_attention_block
        self.cross_attention_block=cross_attention_block
        self.feed_forward=feed_forward_block
        self.residual_connection=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self,x, encoder_output, src_mask, trgt_mask):
         x=self.residual_connection[0](x,lambda x: self.self_attention_block(x,x,x,trgt_mask))
         x=self.residual_connection[1](x,lambda x:self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
         x=self.residual_connection[2](x,self.feed_forward)
         return x

class Decoder(nn.Module):

    def __init__(self, layers:nn.ModuleList):
            super().__init__()
            self.layers=layers
            self.norm=LayerNormalization()
    
    def forward(self,x, encoder_output, src_mask,trgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output, src_mask, trgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.projection_layer=nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.projection_layer(x), dim=-1)

class Transformer(nn.Module):

    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed:InputEmbedding, trgt_embed:InputEmbedding, src_pos:PositionalEmbedding, trgt_pos:PositionalEmbedding, projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.trgt_embed=trgt_embed
        self.src_pos=src_pos
        self.trgt_pos=trgt_pos
        self.projection_layer=projection_layer

    def encode(self, src, src_mask):
        src-self.src_embed(src)
        src=self.src_pos(src)
        return self.encoder(src)

    def decode(self, encoder_output, src_mask, trgt,trgt_mask):
        trgt=self.trgt_embed(trgt)
        trgt=self.trgt_pos(trgt)
        return self.decode(trgt, encoder_output, src_mask, trgt_mask)

    def project(self,x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size, trgt_vocab_size, src_seq_len, trgt_seq_len, d_model=512, N=6, h=8, dropout=0.1, d_ff=2048):
    src_embed=InputEmbedding(d_model, src_vocab_size)
    trgt_embed=InputEmbedding(d_model, trgt_vocab_size)

    src_pos=PositionalEmbedding(d_model, src_seq_len, dropout)
    trgt_pos=PositionalEmbedding(d_model,trgt_seq_len, dropout)

    #Encoder    
    encoder_self_attention_block=MultiHeadAttention(d_model, h, dropout)
    encoder_feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
    encoder_block=EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
    encoder_blocks=[encoder_block for i in range(N)]

    #Decoder
    decoder_self_attention=MultiHeadAttention(d_model, h, dropout)
    decoder_cross_attention=MultiHeadAttention(d_model, h, dropout)
    decoder_feed_forward_block=FeedForwardBlock(d_model, d_ff, dropout)
    decoder_block=DecoderBlock(decoder_self_attention, decoder_cross_attention, decoder_feed_forward_block, dropout)
    decoder_blocks=[decoder_block for i in range(N)]

    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))

    projection_layer=ProjectionLayer(d_model,trgt_vocab_size)

    transformer=Transformer(encoder, decoder, src_embed, trgt_embed, src_pos, trgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return transformer














    


    
