import torch
import torch.nn as nn
import transformer as tr
import data_gen
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import pickle


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 0 : pad, 1: Start of Sentence, 2: End of sentence
# Those lists of token_ids are created by the tokenizer
N = 8192 * 2
N_Val = 128
batch_size = 64 #essayer 1024, 512, 256, 128 dans cet ordre -> ça ralentit l'entrainement sans l'améliorer

generating = False

if generating :
    x, trg, cat_trg = data_gen.train_radom_pattern(N, N_Val, size = 10, nb_concat = 8)
    with open('data_8192_16_x', 'wb') as fichier:
        pickler = pickle.Pickler(fichier)
        pickler.dump(x)
    with open('data_8192_16_trg', 'wb') as fichier:
        pickler = pickle.Pickler(fichier)
        pickler.dump(trg)
    with open('data_8192_16_cat_trg', 'wb') as fichier:
        pickler = pickle.Pickler(fichier)
        pickler.dump(cat_trg)
else :
    print("loading x...")
    with open('data_16384_8_x', 'rb') as fichier:
        unpickler = pickle.Unpickler(fichier)
        x = unpickler.load()
    print("loading trg...")
    with open('data_16384_8_trg', 'rb') as fichier:
        unpickler = pickle.Unpickler(fichier)
        trg = unpickler.load()
    print("loading categorical...")
    with open('data_16384_8_cat_trg', 'rb') as fichier:
        unpickler = pickle.Unpickler(fichier)
        cat_trg = unpickler.load()

x_loader = torch.utils.data.DataLoader(x[:N], batch_size=batch_size, shuffle = False, drop_last=True)
trg_loader = torch.utils.data.DataLoader(trg[:N], batch_size=batch_size, shuffle = False, drop_last=True)
cat_loader = torch.utils.data.DataLoader(cat_trg[:N], batch_size=batch_size, shuffle = False, drop_last=True)

val_x_loader = torch.utils.data.DataLoader(x[N:], batch_size=N_Val, shuffle = False, drop_last=True)
val_trg_loader = torch.utils.data.DataLoader(trg[N:], batch_size=N_Val, shuffle = False, drop_last=True)
val_cat_loader = torch.utils.data.DataLoader(cat_trg[N:], batch_size=N_Val, shuffle = False, drop_last=True)

src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 100
trg_vocab_size = 100

model = tr.Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device, embed_size = 64, num_encoder = 0, num_decoder = 2, heads = 2, max_length = len(x[0])).to(
    device
)
opt = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fct = torch.nn.CrossEntropyLoss()

# Why do we need :-1 here?
epochs = 50
x_plot = []
y_tra_plot = []
y_val_plot = []
y_val_acc  = []

for epoch in range(1, epochs + 1):
    print("epoch : ", epoch)
    print("\ttraining :")
    for batch, batch_cat, batch_trg in zip(x_loader, cat_loader, trg_loader):
      opt.zero_grad()
      pred = model(batch, batch_trg[:, :-1])
      loss = loss_fct(pred, batch_cat[:, 1:])

      if len(x_plot) == 0 :
        x_plot.append(0)
        y_tra_plot.append(loss.item())

      loss.backward()
      opt.step()

      print("#", end = "")

    print("\nloss :", loss.item())
    x_plot.append(epoch)
    y_tra_plot.append(loss.item())

    print("\n\tvalidating :")
    for batch, batch_cat, batch_trg in zip(val_x_loader, val_cat_loader, val_trg_loader):
      pred = model(batch, batch_trg[:, :-1])
      loss = loss_fct(pred, batch_cat[:, 1:])

      y_val_plot.append(loss.item())
      print("loss :", loss.item())

      accuracy = 0
      ones = torch.zeros((pred.shape[0], pred.shape[1]))

      ones[torch.argmax(pred, dim = 2) == batch_trg[:, 1:]] = 1
      accuracy = torch.sum(ones).item() / (pred.shape[0] * pred.shape[1])
      print("accuracy :", accuracy)
      y_val_acc.append(accuracy * y_tra_plot[0])

    if accuracy > 0.46 :
        break
    print("\n\n")

plt.xlim(0, epochs)
plt.ylim(0, y_tra_plot[0])
plt.plot(x_plot, y_tra_plot)
plt.plot(x_plot[1:], y_val_plot)
plt.plot(x_plot[1:], y_val_acc)

x, trg, cat_trg = data_gen.train_fixed_freq_v1(1, 0)

pred = model(x, trg[:, :-1])

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


plt.show()