import torch
import torch.nn as nn
import TransformerGPT as tr
import data_gen
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def lm_cross_entropy_loss(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()



# 0 : pad, 1: Start of Sentence, 2: End of sentence
epochs = 10
N = 8192
N_Val = 1024
f = 30
batch_size = 64
lr = 1e-3
weight_decay = 1e-2


nb_tests = 10
for k in range(nb_tests):
	x = data_gen.train_GPT_induction(N, N_Val, f, only_max = True)
	x_loader = torch.utils.data.DataLoader(x[:N], batch_size=batch_size, shuffle = False, drop_last=True)
	val_x_loader = torch.utils.data.DataLoader(x[N:], batch_size=N_Val, shuffle = False, drop_last=True)

	model_cfg = tr.Config(
		debug=False,
		d_residual=64,
		n_heads=1,
		d_head=16,
		n_layers=2,
		max_length=len(x[0]),
		d_vocab=100
	)

	model = tr.Transformer(model_cfg).to(device)

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


	##### attention diagram #####

	x = data_gen.train_GPT_induction(1, 0, f, only_max = True)

	pred = model(x)[:, :-1]

	accuracy = 0
	ones = torch.zeros((pred.shape[0], pred.shape[1]))

	ones[torch.argmax(pred, dim = 2) == x[:, 1:]] = 1
	accuracy = torch.sum(ones).item() / (pred.shape[0] * pred.shape[1])
	print(accuracy)


	i = 0
	j = 0
	for layer in model.blocks:
	  for head in layer.attn.attention[0]:

	    res = head.cpu().detach()
	    res = np.array(res)
	    res = np.uint8(res * 255 / np.sqrt(np.max(res, axis = 1)))
	    res = Image.fromarray(res)
	    res.save("head_" + str(k) + "_" + str(i) + "_" + str(j) + '.png', 'png')

	    j += 1
	  j = 0
	  i += 1