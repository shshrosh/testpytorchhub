import torch
import torch.nn as nn


class DeepQDSModel(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(vocab_size,emb_size)

        self.mlp = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
        )


class DeepQDSBertModel(nn.Module):
    def __init__(self, dropout_prob, hidden_size,init_bert_weights):
        super().__init__()

        self.batch_counter = 0

        self.dropout = nn.Dropout(dropout_prob)

        self.title_mlp = nn.Linear(hidden_size, 100)
        self.title_mlp_2 = nn.Linear(100, 1)

        self.url_mlp = nn.Linear(hidden_size, 100)
        self.url_mlp.apply(init_bert_weights)
        self.url_mlp_2 = nn.Linear(100, 1)
        self.url_mlp_2.apply(init_bert_weights)

        self.clickstream_mlp = nn.Linear(hidden_size, 100)
        self.clickstream_mlp.apply(init_bert_weights)
        self.clickstream_mlp_2 = nn.Linear(100, 1)
        self.clickstream_mlp_2.apply(init_bert_weights)

        self.orig_url_mlp = nn.Linear(hidden_size, 100)
        self.orig_url_mlp.apply(init_bert_weights)
        self.orig_url_mlp_2 = nn.Linear(100, 1)
        self.orig_url_mlp_2.apply(init_bert_weights)


        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()




    
