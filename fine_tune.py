from transformers import GPT2Model, GPT2Tokenizer

import torch
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import TransformerGPT_with_MLP as tgm
import data_gen as dg

"""
previoous token head :
2:2     2:9     4:11

dupplicate token head :
0:1     0:10    3:0

induction head :
5:5     5:8     5:9     6:9

S-inhibition head :
7:3     7:9     8:6     8:10
"""

def save_attention(model, x, fine_tuned = False):
    model(x)

    h22 = model.blocks[2].attn.attention[0][2].cpu().detach().numpy()
    h22 = np.uint8(h22 * 255 / np.max(h22, axis = 1))
    h22 = Image.fromarray(h22)
    h22.save("gpt2" + ("_ft" if fine_tuned else "") + "_pre_2_2.png", "png")

    h29 = model.blocks[2].attn.attention[0][9].cpu().detach().numpy()
    h29 = np.uint8(h29 * 255 / np.max(h29, axis = 1))
    h29 = Image.fromarray(h29)
    h29.save("gpt2" + ("_ft" if fine_tuned else "") + "_pre_2_9.png", "png")

    h411 = model.blocks[4].attn.attention[0][11].cpu().detach().numpy()
    h411 = np.uint8(h411 * 255 / np.max(h411, axis = 1))
    h411 = Image.fromarray(h411)
    h411.save("gpt2" + ("_ft" if fine_tuned else "") + "_pre_4_11.png", "png")

    h01 = model.blocks[0].attn.attention[0][1].cpu().detach().numpy()
    h01 = np.uint8(h01 * 255 / np.max(h01, axis = 1))
    h01 = Image.fromarray(h01)
    h01.save("gpt2" + ("_ft" if fine_tuned else "") + "_dup_0_1.png", "png")

    h010 = model.blocks[0].attn.attention[0][10].cpu().detach().numpy()
    h010 = np.uint8(h010 * 255 / np.max(h010, axis = 1))
    h010 = Image.fromarray(h010)
    h010.save("gpt2" + ("_ft" if fine_tuned else "") + "_dup_0_10.png", "png")

    h30 = model.blocks[3].attn.attention[0][0].cpu().detach().numpy()
    h30 = np.uint8(h30 * 255 / np.max(h30, axis = 1))
    h30 = Image.fromarray(h30)
    h30.save("gpt2" + ("_ft" if fine_tuned else "") + "_dup_3_0.png", "png")

    h55 = model.blocks[5].attn.attention[0][5].cpu().detach().numpy()
    h55 = np.uint8(h55 * 255 / np.max(h55, axis = 1))
    h55 = Image.fromarray(h55)
    h55.save("gpt2" + ("_ft" if fine_tuned else "") + "_ind_5_5.png", "png")

    h58 = model.blocks[5].attn.attention[0][8].cpu().detach().numpy()
    h58 = np.uint8(h58 * 255 / np.max(h58, axis = 1))
    h58 = Image.fromarray(h58)
    h58.save("gpt2" + ("_ft" if fine_tuned else "") + "_ind_5_8.png", "png")

    h59 = model.blocks[5].attn.attention[0][9].cpu().detach().numpy()
    h59 = np.uint8(h59 * 255 / np.max(h59, axis = 1))
    h59 = Image.fromarray(h59)
    h59.save("gpt2" + ("_ft" if fine_tuned else "") + "_ind_5_9.png", "png")

    h69 = model.blocks[6].attn.attention[0][9].cpu().detach().numpy()
    h69 = np.uint8(h69 * 255 / np.max(h69, axis = 1))
    h69 = Image.fromarray(h69)
    h69.save("gpt2" + ("_ft" if fine_tuned else "") + "_ind_6_9.png", "png")

    h73 = model.blocks[7].attn.attention[0][3].cpu().detach().numpy()
    h73 = np.uint8(h73 * 255 / np.max(h73, axis = 1))
    h73 = Image.fromarray(h73)
    h73.save("gpt2" + ("_ft" if fine_tuned else "") + "_sin_7_3.png", "png")

    h79 = model.blocks[7].attn.attention[0][9].cpu().detach().numpy()
    h79 = np.uint8(h79 * 255 / np.max(h79, axis = 1))
    h79 = Image.fromarray(h79)
    h79.save("gpt2" + ("_ft" if fine_tuned else "") + "_sin_7_9.png", "png")

    h86 = model.blocks[8].attn.attention[0][6].cpu().detach().numpy()
    h86 = np.uint8(h86 * 255 / np.max(h86, axis = 1))
    h86 = Image.fromarray(h86)
    h86.save("gpt2" + ("_ft" if fine_tuned else "") + "_sin_8_6.png", "png")

    h810 = model.blocks[8].attn.attention[0][10].cpu().detach().numpy()
    h810 = np.uint8(h810 * 255 / np.max(h810, axis = 1))
    h810 = Image.fromarray(h810)
    h810.save("gpt2" + ("_ft" if fine_tuned else "") + "_sin_8_10.png", "png")


def lm_cross_entropy_loss(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()


# Load the GPT2Small model from the transformers library
gpt2small = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

cfg = tgm.Config()
cfg.d_residual = 768
cfg.debug = False
cfg.layer_norm_eps = 1e-5
cfg.d_vocab = 50257
cfg.init_range = 0.02
cfg.max_length = 1024
cfg.d_head = 64
cfg.n_heads = 12
cfg.n_layers = 12

model = tgm.Transformer(cfg)

# Load the weights and biases for the embedding layer
print("Load the weights and biases for the embedding layer...", end="")
w = gpt2small.wte.weight.data
model.embed.W_E.data = w
print(" done.")

# Load the weights and biases for the positional embedding layer
print("Load the weights and biases for the positional embedding layer...", end="")
w = gpt2small.wpe.weight.data
model.pos_embed.W_pos.data = w
print(" done.")

print("Load the weights and biases for the layer norms and attention layers...", end="")
for layer in range(cfg.n_layers):
    # Load the weights and biases for the first layer norm
    w = gpt2small.h[layer].ln_1.weight.data
    b = gpt2small.h[layer].ln_1.bias.data
    model.blocks[layer].ln1.w.data = w
    model.blocks[layer].ln1.b.data = b

    # Load the weights and biases for the attention layer
    w = gpt2small.h[layer].attn.c_attn.weight.data
    q, k, v = torch.chunk(w, 3, dim=-1)
    b = gpt2small.h[layer].attn.c_attn.bias.data
    q_bias, k_bias, v_bias = torch.chunk(b, 3, dim=-1)

    w_0 = gpt2small.h[layer].attn.c_proj.weight.data
    b_0 = gpt2small.h[layer].attn.c_proj.bias.data
    for head in range(cfg.n_heads):
        model.blocks[layer].attn.W_Q[head].data = q[head::cfg.n_heads]
        model.blocks[layer].attn.W_K[head].data = k[head::cfg.n_heads]
        model.blocks[layer].attn.W_V[head].data = v[head::cfg.n_heads]

        model.blocks[layer].attn.b_Q[head].data = q_bias[head::cfg.n_heads]
        model.blocks[layer].attn.b_K[head].data = k_bias[head::cfg.n_heads]
        model.blocks[layer].attn.b_V[head].data = v_bias[head::cfg.n_heads]

        model.blocks[layer].attn.W_O[head].data = w_0[head::cfg.n_heads]
        model.blocks[layer].attn.b_O[head].data = b_0[head::cfg.n_heads]
    
    # Load the weights and biases for the second layer norm
    w = gpt2small.h[layer].ln_2.weight.data
    b = gpt2small.h[layer].ln_2.bias.data
    model.blocks[layer].ln2.w.data = w
    model.blocks[layer].ln2.b.data = b

    # Load the weights and biases for the MLP
    w = gpt2small.h[layer].mlp.c_fc.weight.data
    b = gpt2small.h[layer].mlp.c_fc.bias.data
    model.blocks[layer].mlp.W_in.data = w
    model.blocks[layer].mlp.b_in.data = b

    w = gpt2small.h[layer].mlp.c_proj.weight.data
    b = gpt2small.h[layer].mlp.c_proj.bias.data
    model.blocks[layer].mlp.W_out.data = w
    model.blocks[layer].mlp.b_out.data = b
print(" done.")

# Load the weights and biases for the final layer norm
print("Load the weights and biases for the final layer norm...", end="")
w = gpt2small.ln_f.weight.data
b = gpt2small.ln_f.bias.data
model.ln_final.w.data = w
model.ln_final.b.data = b
print(" done.")

# the unembed layer is the transpose of the embedding layer
# there is no b_U in gpt2small so b_U is set to zero
model.unembed.W_U.data = model.embed.W_E.data.t()
model.unembed.b_U.data = torch.zeros(cfg.d_vocab)

model.to("cuda:0")

#use the GPT2Tokenizer to tokenize the sentence
sentence = "Mr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Mr Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large moustache. Mrs Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbours. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere. The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn't think they could bear it if anyone found out about the Potters. Mrs Potter was Mrs Dursley's sister, but they hadn't met for several years; in fact, Mrs Dursley pretended she didn't have a sister, because her sister and her good- for-nothing husband were as unDursleyish as it was possible to be. The Dursleys shuddered to think what the neighbours would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son, too, but they had never even seen him. This boy was another good reason for keeping the Potters away; they didn't want Dudley mixing with a child like that."
tokens = tokenizer.encode(sentence)
tokens = torch.tensor([tokens]).to("cuda:0")

pred = model(tokens)

save_attention(model, tokens)

#now convert the prediction to a string
print(tokenizer.decode(pred[0].argmax(dim=-1).tolist()))

x_attention, trg = dg.fine_tune_most_frequent(1,0, 60)
save_attention(model, x_attention)

N = 64
N_Val = 16
size = 60

batch_size = 8
epochs = 10
lr = 1e-4
weight_decay = 1e-5

opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

x, trg = dg.fine_tune_most_frequent(N, N_Val, size)
x_loader = torch.utils.data.DataLoader(x[:N], batch_size=batch_size, shuffle=False, drop_last=True)
val_x_loader = torch.utils.data.DataLoader(x[N:], batch_size=N_Val, shuffle=False, drop_last=True)

trg_loader = torch.utils.data.DataLoader(trg[:N], batch_size=batch_size, shuffle=False, drop_last=True)
val_trg_loader = torch.utils.data.DataLoader(trg[N:], batch_size=N_Val, shuffle=False, drop_last=True)

x_plot = []
y_training = []
y_validation = []
y_validation_accuracy = []

for epoch in range(1, epochs + 1):
    print("Epoch: ", epoch)
    print("\tTraining...")

    #TODO : data sur cpu, puis mettre sur gpu en début de boucle et enlever en fin de boucle (pour éviter de surcharger la mémoire)
    for batch, batch_trg in zip(x_loader, trg_loader):
        opt.zero_grad()
        pred = model(batch)
        loss = lm_cross_entropy_loss(pred, batch_trg)
        loss.backward()
        opt.step()

        if len(x_plot) == 0 :
            x_plot.append(0)
            y_training.append(loss.item())
        
        print("#", end="")
    
    print("\nloss: ", loss.item())
    x_plot.append(epoch)
    y_training.append(loss.item())

    print("\tValidation...")
    for batch, batch_trg in zip(val_x_loader, val_trg_loader):
        pred = model(batch)
        loss = lm_cross_entropy_loss(pred, batch_trg)
        
        y_validation.append(loss.item())
        print("loss: ", loss.item())

        accuracy = 0
        ones = torch.zeros(batch_trg.shape).to("cuda:0")

        ones[torch.argmax(pred, dim = 2) == batch_trg] = 1
        accuracy = torch.sum(ones).item() / (batch_trg.shape[0] * batch_trg.shape[1])
        y_validation_accuracy.append(accuracy * y_training[0])
        print("accuracy: ", accuracy)
    
    print("\n\n")

plt.xlim(0, epochs)
plt.ylim(0, y_training[0])

plt.plot(x_plot, y_training, label="Training")
plt.plot(x_plot[1:], y_validation, label="Validation Loss")
plt.plot(x_plot[1:], y_validation_accuracy, label="Validation Accuracy")

plt.legend()
plt.show()

#save the model
#torch.save(model.state_dict(), "model.pt")

save_attention(model, x_attention, fine_tuned=True)