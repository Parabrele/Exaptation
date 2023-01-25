import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.attention = None

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Attention weights of the different heads are stored in the same tensors.

        # We put all the attention heads in the same Linears Layers:
        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)

        # fc_out after having concatenated the results of the attention of each head
        self.fc_out = nn.Linear(self.embed_size, embed_size)

    def forward(self, values, keys, queries, mask):
        """
        values: (N, value_len, embed_size)
        keys: (N, keys_len, embed_size)
        queries: (N, queries_len, embed_size)

        mask: None or (N, heads, query_len, key_len) 
        if mask == 0, attention matrix  -> float("-1e20") (big negative value)
        Ignore the mask, it's too difficult at the beginning.
        """
        # Get number of training examples

        N = queries.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Compute the values keys and queries
        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(queries)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Compute the similarity between query*keys
        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for better stability
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)

        self.attention = attention

        # attention shape: (N, heads, query_len, key_len)


        # We compute the attention (aka the ponderated value)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.


        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)
        return self.fc_out(out)
    
    def forward2(self, values, keys, queries, mask):
        """
        values: (N, value_len, embed_size)
        keys: (N, keys_len, embed_size)
        queries: (N, queries_len, embed_size)

        mask: None or (N, heads, query_len, key_len) 
        if mask == 0, attention matrix  -> float("-1e20") (big negative value)
        Ignore the mask, it's too difficult at the beginning.
        """
        N = queries.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Compute the values keys and queries
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = values.reshape(N, key_len, self.heads, self.head_dim)
        queries = values.reshape(N, query_len, self.heads, self.head_dim)
        


        # Compute the similarity between query*keys
        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        energy = torch.einsum('Nqhd,Nkhd->Nhqk' , [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)


        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))


        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for better stability
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)


        # We compute the attention
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.


        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)
        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
            #ya pas de relu parce qu'on veut réaterrir dans l'espace de sémantique
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        """Reproduce the above figure.
        
        Tip: Dropout is always used after LayerNorm
        """
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
    def forward2(self, value, key, query, mask):
        """Reproduce the above figure.
        
        Tip: Dropout is always used after LayerNorm
        """
        attention = self.attention(value, key, query, mask)
        #cf les notes, query c'est pas la matrice Q, c'est X3 qu'on utilise dans SelfAttention pour calculer la vraie Q. Donc c'est l'entrée par le bas. Qu'on propage, du coup, comme un residual
        x = self.norm1(attention + query)
        x = self.dropout(x)
        y = self.feed_dorward(x)
        y = self.norm2(x + y)
        y = self.dropout(y)
        return y


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x: Tokenized tensor (N, seq_length) containing tokens_ids
        mask: Used for masking the padding inside the encoder.

        Create the position_embedding/word_embedding
        add the embeddings and forward it the all the layers.

        Tip: In order to create the position_embedding, you will need torch.arange and tensor.expand
        """
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        position_embedding = self.position_embedding(positions)
        word_embedding = self.word_embedding(x)
        out = self.dropout(position_embedding + word_embedding)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
    
    def forward2(self, x, mask):
        """
        x: Tokenized tensor (N, seq_length) containing tokens_ids
        mask: Used for masking the padding inside the encoder.

        Create the position_embedding/word_embedding
        add the embeddings and forward it the all the layers.

        Tip: In order to create the position_embedding, you will need torch.arange and tensor.expand
        """
        N, seq_length = x.shape

        time = torch.arange(0, seq_length)
        time = time.expand(N, seq_length).to(self.device)
        time_embed = self.position_embedding(time)
        x_embed = self.word_embedding(x)
        z = self.dropout(x_embed + time_embed)

        for layer in self.layers :
          z = layer(z, z, z, mask)
        
        return z


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        """DecoderBlock = masked multi-head attention + TransformerBlock"""
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
    
    def forward2(self, x, value, key, src_mask, trg_mask):
        """DecoderBlock = masked multi-head attention + TransformerBlock"""
        #c'est en gros un trans_block tronqué puis un vrai
        attention = self.attention(x, x, x, trg_mask)
        x_q = self.dropout(self.norm(attention + x))
        z = self.transformer_block(value, key, x_q, src_mask)

        return z


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        """Same as Encoder"""
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out

    def forward2(self, x, enc_out, src_mask, trg_mask):
        """Same as Encoder"""
        N, seq_length = x.shape

        time = torch.arange(0, seq_length)
        time = time.expand(N, seq_length).to(self.device)
        time_embed = self.position_embedding(time)
        x_embed = self.word_embedding(x)
        z = self.dropout(x_embed + time_embed)
        
        for layer in self.layers :
          z = layer(z, enc_out, enc_out, src_mask, trg_mask)
        
        z = self.fc_out(z)
        
        return z


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        num_encoder=None,
        num_decoder=None,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers if num_encoder is None else num_encoder,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers if num_decoder is None else num_decoder,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """src is a tensor containing sequences of tokens. Some sequences have been padded.
        
        The purpose of the src_mask is to mask those padded tokens.
        This mask is used both during training and inference time.
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    
    def make_trg_mask(self, trg):
        """trg is a tensor containing sequences of tokens which have been predicted.

        trg mask is used only during training.
        """
        # Bonus: Why do we use a lower triangular matrix?
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
