import torch
import torch.nn as nn
import TransformerGPT as tr
import data_gen
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import pickle



def lm_cross_entropy_loss(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()



# 0 : pad, 1: Start of Sentence, 2: End of sentence
epochs = 10
N = 8192
N_Val = 64
f = 10
batch_size = 64
lr = 1e-3
weight_decay = 1e-2

x = data_gen.train_GPT_induction(N, N_Val, f)
x_loader = torch.utils.data.DataLoader(x[:N], batch_size=batch_size, shuffle = False, drop_last=True)
val_x_loader = torch.utils.data.DataLoader(x[N:], batch_size=N_Val, shuffle = False, drop_last=True)

model_cfg = tr.Config(debug=False, d_residual=256, n_heads=2, d_head=32, n_layers=2, max_length=len(x[0]), d_vocab=100)

model = tr.Transformer(model_cfg)

opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

x_plot = []
y_tra_plot = []
y_val_plot = []
y_val_acc  = []

for epoch in range(1, epochs + 1):
    print("epoch : ", epoch)
    print("\ttraining :")
    for batch in x_loader:
      pred = model(batch)
      loss = lm_cross_entropy_loss(pred, batch)
      loss.backward()
      opt.step()
      opt.zero_grad()

      if len(x_plot) == 0 :
        x_plot.append(0)
        y_tra_plot.append(loss.item())


      print("#", end = "")

    print("\nloss :", loss.item())
    x_plot.append(epoch)
    y_tra_plot.append(loss.item())

    print("\n\tvalidating :")
    for batch in val_x_loader:
      pred = model(batch)
      loss = lm_cross_entropy_loss(pred, batch)

      y_val_plot.append(loss.item())
      print("loss :", loss.item())

      accuracy = 0
      ones = torch.zeros((pred.shape[0], pred.shape[1]-1))

      ones[torch.argmax(pred[:, :-1], dim = 2) == batch[:, 1:]] = 1
      accuracy = torch.sum(ones).item() / (pred.shape[0] * pred.shape[1])
      print("accuracy :", accuracy)
      y_val_acc.append(accuracy * y_tra_plot[0])

    print("\n\n")

plt.xlim(0, epochs)
plt.ylim(0, y_tra_plot[0])
plt.plot(x_plot, y_tra_plot)
plt.plot(x_plot[1:], y_val_plot)
plt.plot(x_plot[1:], y_val_acc)

plt.show()

"""
x = data_gen.train_GPT_induction()

pred = model(x)[:, :-1]

accuracy = 0
ones = torch.zeros((pred.shape[0], pred.shape[1]))

ones[torch.argmax(pred, dim = 2) == trg[:, :-1]] = 1
accuracy = torch.sum(ones).item() / (pred.shape[0] * pred.shape[1])
print(accuracy)

i = 0
j = 0
for layer in model.encoder.layers:
  for head in layer.attention.attention[0]:
    print(head.shape)
    print(head)

    res = head.cpu().detach()
    res = np.array(res)
    res = np.uint8(res / np.sqrt(np.max(res)) * 255)
    res = Image.fromarray(res)
    res.save("head_" + str(i) + "_" + str(j) + '.png', 'png')

    j += 1

i = 1
j = 0
for layer in model.decoder.layers:
  for head in layer.attention.attention[0]:

    res = head.cpu().detach()
    res = np.array(res)
    res = np.uint8(res * 255 / np.sqrt(np.max(res, axis = 1)))
    res = Image.fromarray(res)
    res.save("head_" + str(i) + "_" + str(j) + '.png', 'png')

    j += 1
  j = 0

  for head in layer.transformer_block.attention.attention[0]:

    res = head.cpu().detach()
    res = np.array(res)
    res = np.uint8(res * 255 / (np.max(res)))
    res = Image.fromarray(res)
    res.save("head_" + str(i) + "_" + str(j) + '_bis.png', 'png')

    j += 1
  j = 0
  i += 1


plt.show()"""