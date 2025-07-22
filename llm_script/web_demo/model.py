import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ
import torch
import math
from torch.nn import functional as F


ax_dim = 32
a_dim = 8
x_dim = 8

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        num = int(bool(config.num_props))
        # num = 1
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                                     .view(1, 1, config.block_size + num, config.block_size + num))

        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        y, attn = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)
        if config.num_props:
            self.prop_nn = nn.Linear(config.num_props, config.n_embd)
     
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size

        if config.lstm:
            self.lstm = nn.LSTM(input_size = config.n_embd, hidden_size = config.n_embd, num_layers = config.lstm_layers, dropout = 0.3, bidirectional = False)
        self.apply(self._init_weights)

        #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, prop = None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.config.num_props:
            assert prop.size(-1) == self.config.num_props, "Num_props should be equal to last dim of property vector"           

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        type_embeddings = self.type_emb(torch.ones((b,t), dtype = torch.long, device = idx.device))
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)
        
        embed = x

        if self.config.num_props:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))
            if prop.ndim == 2:
                p = self.prop_nn(prop.unsqueeze(1))    # for single property
            else:
                p = self.prop_nn(prop)    # for multiproperty
            p += type_embd
            x = torch.cat([p, x], 1)

        # x = self.blocks(x)
        attn_maps = []

        for layer in self.blocks:
            x, attn = layer(x)
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)

        if self.config.num_props:
            num = int(bool(self.config.num_props))
        else:
            num = 0

        logits = logits[:, num:, :]

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, attn_maps, embed # (num_layers, batch_size, num_heads, max_seq_len, max_seq_len)
        
        
    @torch.no_grad()
    def sample(self, x, steps, temperature=1.0, do_sample=False, top_k=None, top_p=None, prop=None):
        """
        Take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
        
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        #model.eval()
        
        def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
            """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
                Args:
                    logits: logits distribution shape (batch size x vocabulary size)
                    top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                    top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                        Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
                From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
            """
            top_k = min(top_k, logits.size(-1))  # Safety check
            if top_k > 0:
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = filter_value
        
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
        
                # scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                logits[indices_to_remove] = filter_value
            return logits
        
         
        for k in range(steps):
            x_cond = x if x.size(1) <= self.block_size else x[:, -self.block_size:] # crop context if needed

            # forward the model to get the logits for the index in the sequence
            logits, _, _, _ = self(x_cond, prop = prop) # for sampling, no target

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options OR using nucleus (top-p) filtering
            #if top_k is not None:
            #    v, _ = torch.topk(logits, top_k)
            #    logits[logits < v[:, [-1]]] = -float('Inf')
            logits = top_k_top_p_filtering(logits, top_p=top_p, top_k=top_k)

                
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution or take the most likely
            if do_sample:
                x_next = torch.multinomial(probs, num_samples=1)
            else:
                _, x_next = torch.topk(probs, k=1, dim=-1)

            # append sampled index to the running sequence and continue
            x = torch.cat((x, x_next), dim=1)

        return x[:, 1:]

adj_vec_dim = 225
ptb_vec_dim = 45
latent_dim = 48

class vaeModel(nn.Module):
    def __init__(self):
        super(vaeModel, self).__init__()
        ## encoder layers ##
        # adj_encoder
        self.en_adj_ch1, self.en_adj_ch2, self.en_adj_ch3, self.en_adj_ch4 = 12, 16, 16, 16
        self.en_adj_kernel1, self.en_adj_kernel2, self.en_adj_kernel3, self.en_adj_kernel4 = 3, 3, 2, 2
        self.en_adj_stride1, self.en_adj_stride2, self.en_adj_stride3, self.en_adj_stride4 = 1, 1, 1, 1
        self.en_adj_dim1, self.en_adj_dim2, self.en_adj_dim3, self.en_adj_dim4 = 512, 512, 512, 128

        self.en_x_ch1, self.en_x_ch2 = 20, 36
        self.en_x_kernel1, self.en_x_kernel2 = (3, 1), (2, 1)
        self.en_x_stride1, self.en_x_stride2 = (2, 1), 1
        self.en_x_dim1, self.en_x_dim2 = 12, 1
        self.en_x_hd1, self.en_x_hd2, self.en_x_hd3, self.en_x_hd4 = 256, 256, 128, latent_dim
        
        self.en_adj_fc1 = nn.Linear(adj_vec_dim, self.en_adj_dim1)
        self.en_adj_fc2 = nn.Linear(self.en_adj_dim1, self.en_adj_dim2)
        self.en_adj_fc3 = nn.Linear(self.en_adj_dim2, self.en_adj_dim3)
        self.en_adj_fc4 = nn.Linear(self.en_adj_dim3, self.en_adj_dim4)
        self.en_adj_fc5 = nn.Linear(self.en_adj_dim4, a_dim + ax_dim)

        self.adj_fc_mu = nn.Linear(a_dim + ax_dim,  a_dim + ax_dim)
        self.adj_fc_std = nn.Linear(a_dim + ax_dim, a_dim + ax_dim)

        # x_encoder
        x_hidden_dim = [ptb_vec_dim, 640, 640, 640, 512, ax_dim + x_dim]
        
        self.en_x_fc1 = nn.Linear(x_hidden_dim[0], x_hidden_dim[1])
        self.en_x_fc2 = nn.Linear(x_hidden_dim[1], x_hidden_dim[2])
        self.en_x_fc3 = nn.Linear(x_hidden_dim[2], x_hidden_dim[3])
        self.en_x_fc4 = nn.Linear(x_hidden_dim[3], x_hidden_dim[4])
        self.en_x_fc5 = nn.Linear(x_hidden_dim[4], x_hidden_dim[5])
   
        self.x_fc_mu = nn.Linear(ax_dim + x_dim,  ax_dim + x_dim)
        self.x_fc_std = nn.Linear(ax_dim + x_dim, ax_dim + x_dim)

        # adj_decoder
        self.adj_de_fc = nn.Linear(24, self.en_x_hd3)
        self.x_de_fc = nn.Linear(24, self.en_x_hd3)

        self.de_adj_fc1 = nn.Linear(a_dim + ax_dim, self.en_adj_dim4)
        self.de_adj_fc2 = nn.Linear(self.en_adj_dim4, self.en_adj_dim3)
        self.de_adj_fc3 = nn.Linear(self.en_adj_dim3, self.en_adj_dim2)
        self.de_adj_fc4 = nn.Linear(self.en_adj_dim2, self.en_adj_dim1)
        self.de_adj_fc5 = nn.Linear(self.en_adj_dim1, adj_vec_dim)
       
        # x_decoder
        de_x_hidden_dim = [x_dim + ax_dim, 640, 640, 512, 256, ptb_vec_dim]
        self.de_x_fc1 = nn.Linear(de_x_hidden_dim[0], de_x_hidden_dim[1])
        self.de_x_fc2 = nn.Linear(de_x_hidden_dim[1], de_x_hidden_dim[2])
        self.de_x_fc3 = nn.Linear(de_x_hidden_dim[2], de_x_hidden_dim[3])
        self.de_x_fc4 = nn.Linear(de_x_hidden_dim[3], de_x_hidden_dim[4])
        self.de_x_fc5 = nn.Linear(de_x_hidden_dim[4], de_x_hidden_dim[5])

        self.activation = nn.ReLU()

        self.residual_vq = ResidualVQ(
            dim = 48,
            codebook_size = 256,
            num_quantizers = 16,
            kmeans_init = True,
            kmeans_iters = 40
        )

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(logvar)
        eps = torch.FloatTensor(logvar.size()[0],1).normal_(0,1)
        eps = eps.to(mu.device)
        eps  = eps.expand(sigma.size())
        return mu + sigma*eps

    def encoder(self, adj, x):
        # Adjacency matrix dimensions: -> 512 -> 512 -> 512 -> 128 -> 8 + 32
        
        adj = self.activation(self.en_adj_fc1(adj))
        adj = self.activation(self.en_adj_fc2(adj))
        adj = self.activation(self.en_adj_fc3(adj))
        adj = self.activation(self.en_adj_fc4(adj))
        adj = self.en_adj_fc5(adj)

        # Property dimensions: -> 640 -> 640 -> 640 -> 512 -> 8 + 32
        # mu: 8 + 32, std: 8 + 32
        x = self.activation(self.en_x_fc1(x))
        x = self.activation(self.en_x_fc2(x))
        x = self.activation(self.en_x_fc3(x))
        x = self.activation(self.en_x_fc4(x))
        x = self.en_x_fc5(x)

        # mu: 8 + 32, std: 8 + 32
        adj_mu = self.adj_fc_mu(adj)
        adj_log_var = self.adj_fc_std(adj)

        adj_z = self.reparameterize(adj_mu, adj_log_var)
        # mu: 8 + 32, std: 8 + 32
        x_mu = self.x_fc_mu(x)
        x_log_var = self.x_fc_std(x)

        x_z = self.reparameterize(x_mu, x_log_var)
        mu_tot = torch.cat((adj_mu, x_mu), dim = -1)
        log_var_tot = torch.cat((adj_log_var, x_log_var), dim = -1)
        mu = torch.zeros([len(mu_tot), a_dim + ax_dim + x_dim]).to(x.device)
        log_var = torch.zeros([len(mu_tot), a_dim + ax_dim + x_dim]).to(x.device)
        mu[:,:a_dim] = mu_tot[:,:a_dim]
        mu[:,a_dim:(a_dim + ax_dim)] = 0.5*(mu_tot[:,a_dim:a_dim + ax_dim] + mu_tot[:,a_dim + ax_dim:a_dim+2*ax_dim])
        mu[:,a_dim+ax_dim:] = mu_tot[:,a_dim+2*ax_dim:]

        log_var[:,:a_dim] = log_var_tot[:,:a_dim]
        log_var[:,a_dim:(a_dim + ax_dim)] = 0.5*(log_var_tot[:,a_dim:a_dim + ax_dim] + log_var_tot[:,a_dim + ax_dim:a_dim+2*ax_dim])
        log_var[:,a_dim+ax_dim:] = log_var_tot[:,a_dim+2*ax_dim:]

        z = self.reparameterize(mu, log_var)

        return z, mu, log_var
    
    def quantize(self, x):
        quantized, indices, commit_loss = self.residual_vq(x)
        return quantized, indices
    
    def recover(self, indices):
        quantized = self.residual_vq.get_output_from_indices(indices)
        return quantized
    
    def decoder(self, z):

        adj_de_input = z[:,:a_dim+ax_dim]
        x_de_input = z[:,a_dim:]

        adj = self.activation(self.de_adj_fc1(adj_de_input))
        adj = self.activation(self.de_adj_fc2(adj))
        adj = self.activation(self.de_adj_fc3(adj))
        adj = self.activation(self.de_adj_fc4(adj))
        adj = self.de_adj_fc5(adj)
        
        x = self.activation(self.de_x_fc1(x_de_input))
        x = self.activation(self.de_x_fc2(x))
        x = self.activation(self.de_x_fc3(x))
        x = self.activation(self.de_x_fc4(x))
        x = self.de_x_fc5(x)
        adj_out = torch.sigmoid(adj)
        x_out = torch.sigmoid(x)
        return adj_out, x_out

c_hidden_dim = [latent_dim, 400, 800, 1000, 400, 400, 200]
paramDim = 12

class cModel(nn.Module):
    def __init__(self):
        super(cModel, self).__init__()
        self.dropout_p = 0.

        self.model = torch.nn.Sequential()
        self.model.add_module('e_fc1', nn.Linear(a_dim + ax_dim + x_dim, c_hidden_dim[0], bias = False))
        self.model.add_module('e_relu1',nn.ReLU())
        for i in range(1, len(c_hidden_dim)):
            self.model.add_module('e_fc' + str(i+1),nn.Linear(c_hidden_dim[i - 1], c_hidden_dim[i], bias = False))
            self.model.add_module('e_relu' + str(i+1),nn.ReLU())
            self.model.add_module('e_dropout' + str(i+1), nn.Dropout(self.dropout_p))
        self.model.add_module('de_out',nn.Linear(c_hidden_dim[-1], paramDim, bias = False))

    def forward(self, z):
        out = self.model(z)
        return out

def kld_loss(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            # m.bias.data.fill_(0.01)