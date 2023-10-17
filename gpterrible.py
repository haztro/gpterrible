import numpy as np
import math
import json

block_size = 256
eval_iters = 200
n_embd = 384
n_head = 6
head_size = n_embd // n_head
n_layer = 6


def dict_to_array(d):
    return np.array(d['data']).reshape(d['shape'])


def load_tokenizer():
    with open('tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer_dict = json.load(f)

    stoi = {k: int(v) for k, v in tokenizer_dict['stoi'].items()}
    itos = {int(k): v for k, v in tokenizer_dict['itos'].items()}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode


def load_model():
    with open('model.json', 'r') as f:
        json_model = json.load(f)

    model = {'tok_emb': [], 'pos_emb': [], 'blocks': [], 'ln_f': [], 'lm_head': []}
    model['tok_emb'].append(dict_to_array(json_model['tok_emb'][0]))
    model['pos_emb'].append(dict_to_array(json_model['pos_emb'][0]))

    for i in range(len(json_model['blocks'])):
        block = {'multihead': {'heads': [], 'proj': []}, 'ffwd': [], 'ln1': [], 'ln2': []}
        for j in range(len(json_model['blocks'][i]['multihead']['heads'])):
            head = {}
            head['key'] = dict_to_array(json_model['blocks'][i]['multihead']['heads'][j]['key'][0])
            head['query'] = dict_to_array(json_model['blocks'][i]['multihead']['heads'][j]['query'][0])
            head['value'] = dict_to_array(json_model['blocks'][i]['multihead']['heads'][j]['value'][0])
            block['multihead']['heads'].append(head)
        block['multihead']['proj'].append(dict_to_array(json_model['blocks'][i]['multihead']['proj'][0]))
        block['multihead']['proj'].append(dict_to_array(json_model['blocks'][i]['multihead']['proj'][1]))
        block['ffwd'].append(dict_to_array(json_model['blocks'][i]['ffwd'][0]))
        block['ffwd'].append(dict_to_array(json_model['blocks'][i]['ffwd'][1]))
        block['ffwd'].append(dict_to_array(json_model['blocks'][i]['ffwd'][2]))
        block['ffwd'].append(dict_to_array(json_model['blocks'][i]['ffwd'][3]))
        block['ln1'].append(dict_to_array(json_model['blocks'][i]['ln1'][0]))
        block['ln1'].append(dict_to_array(json_model['blocks'][i]['ln1'][1]))
        block['ln2'].append(dict_to_array(json_model['blocks'][i]['ln2'][0]))
        block['ln2'].append(dict_to_array(json_model['blocks'][i]['ln2'][1]))
        model['blocks'].append(block)

    model['ln_f'].append(dict_to_array(json_model['ln_f'][0]))
    model['ln_f'].append(dict_to_array(json_model['ln_f'][1]))
    model['lm_head'].append(dict_to_array(json_model['lm_head'][0]))
    model['lm_head'].append(dict_to_array(json_model['lm_head'][1]))

    return model



def forward(x):
    tok_emb = model['tok_emb'][0][x.astype(int)] # (B, T, C)
    pos_emb = model['pos_emb'][0][np.arange(x.shape[1])] # (T, C)
    x = tok_emb + pos_emb # (B, T, C)

    # Forward each block
    for i in range(n_layer):
        # First Layer Norm
        ln1_weight = model['blocks'][i]['ln1'][0] # (C,)
        ln1_bias = model['blocks'][i]['ln1'][1] # (C,)
        mean = np.mean(x, axis=-1, keepdims=True) # (B, T, 1)
        var = np.var(x, axis=-1, keepdims=True, ddof=1) # (B, T, 1)
        x_norm = (x - mean) / np.sqrt(var + 1e-5) # (B, T, C)
        ln1_out = ln1_weight * x_norm + ln1_bias # (B, T, C)

        # Forward each attention head
        concat = []
        for j in range(n_head):
            B, T, C = ln1_out.shape
            key_weight = model['blocks'][i]['multihead']['heads'][j]['key'] # (C, hs)
            query_weight = model['blocks'][i]['multihead']['heads'][j]['query'] # (C, hs)
            value_weight = model['blocks'][i]['multihead']['heads'][j]['value'] # (C, hs)
            tril = np.tril(np.ones((block_size, block_size))) # (T, T)

            key = ln1_out @ key_weight.T # (B, T, hs)
            query = ln1_out @ query_weight.T # (B, T, hs)
            wei = query @ key.transpose(0, 2, 1) * key.shape[-1]**-0.5 # (B, T, T)
            wei[0][tril[:T, :T] == 0] = np.NINF # (B, T, T)
            wei = np.exp(wei) / np.sum(np.exp(wei), axis=-1, keepdims=True) # (B, T, T)

            value = ln1_out @ value_weight.T # (B, T, hs)
            out = wei @ value # (B, T, hs)
            concat.append(out) 

        concat_heads = np.concatenate(concat, axis=-1) # (B, T, C)

        proj_weight = model['blocks'][i]['multihead']['proj'][0] # (hs * n_head, C)
        proj_bias = model['blocks'][i]['multihead']['proj'][1] # (C,)
        multihead = concat_heads @ proj_weight.T + proj_bias # (B, T, C)
        x = x + multihead # (B, T, C)

        ln2_weight = model['blocks'][i]['ln2'][0] # (C,)
        ln2_bias = model['blocks'][i]['ln2'][1] # (C,)
        mean = np.mean(x, axis=-1, keepdims=True) # (B, T, 1)
        var = np.var(x, axis=-1, keepdims=True, ddof=1) # (B, T, 1)
        x_norm = (x - mean) / np.sqrt(var + 1e-5) # (B, T, C)
        ln2_out = ln2_weight * x_norm + ln2_bias # (B, T, C)

        ffwd_weight1 = model['blocks'][i]['ffwd'][0] # (C, 4 * C)
        ffwd_bias1 = model['blocks'][i]['ffwd'][1] # (C,)
        ffwd_weight2 = model['blocks'][i]['ffwd'][2] # (4 * C, C)
        ffwd_bias2 = model['blocks'][i]['ffwd'][3] # (C,)
        ffwdl1_out = ln2_out @ ffwd_weight1.T + ffwd_bias1 # (B, T, 4 * C)
        relu_out = np.maximum(np.zeros_like(ffwdl1_out), ffwdl1_out) # (B, T, 4 * C)
        ffwd_out = relu_out @ ffwd_weight2.T + ffwd_bias2 # (B, T, C)
        
        x = x + ffwd_out # (B, T, C)

    ln_f_weight = model['ln_f'][0] # (C,)
    ln_f_bias = model['ln_f'][1] # (C,)
    mean = np.mean(x, axis=-1, keepdims=True) # (B, T, 1)
    var = np.var(x, axis=-1, keepdims=True, ddof=1) # (B, T, 1)
    x_norm = (x - mean) / np.sqrt(var + 1e-5) # (B, T, C)
    x = ln_f_weight * x_norm + ln_f_bias # (B, T, C)

    lm_head_weight = model['lm_head'][0] # (C, vocab_size)
    lm_head_bias = model['lm_head'][1] # (vocab_size,)
    logits = x @ lm_head_weight.T + lm_head_bias # (B, T, vocab_size)
    return logits


def generate(model, decode, context, max_tokens):
    for _ in range(max_tokens):
        context_cond = context[:, -block_size:]
        logits = forward(context_cond)
        logits = logits[:, -1, :] # (B, C)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True) # (B, T, T)
        context_next = np.random.choice(range(probs.shape[1]), size=1, p=probs.ravel())
        context = np.concatenate((context, [context_next]), axis=1) # (B, T+1)

        print(decode(context_next.tolist()), end='', flush=True)


model = load_model()
encode, decode = load_tokenizer()
context = np.zeros((1, 1))
generate(model, decode, context, 500)