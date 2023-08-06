from transformers import GPT2Model, GPT2Tokenizer

import torch
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import TransformerGPT_with_MLP as tgm
import data_gen as dg

from tqdm import tqdm

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


def lm_cross_entropy_loss(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()

def fine_tune(model, lr = 1e-4, weight_decay = 1e-5, epochs = 10, batch_size = 8, N = 64, N_Val = 16, size = 60, verbose = True):
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

    for epoch in tqdm(range(1, epochs + 1)):
        if verbose :
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
            
            if verbose :
                print("#", end="")
        
        if verbose :
            print("\nloss: ", loss.item())

        x_plot.append(epoch)
        y_training.append(loss.item())

        if verbose :
            print("\tValidation...")
        for batch, batch_trg in zip(val_x_loader, val_trg_loader):
            pred = model(batch)
            loss = lm_cross_entropy_loss(pred, batch_trg)
            
            y_validation.append(loss.item())
            if verbose :
                print("loss: ", loss.item())

            accuracy = 0
            ones = torch.zeros(batch_trg.shape).to("cuda:0")

            ones[torch.argmax(pred, dim = 2) == batch_trg] = 1
            accuracy = torch.sum(ones).item() / (batch_trg.shape[0] * batch_trg.shape[1])
            y_validation_accuracy.append(accuracy)
            if verbose :
                print("accuracy: ", accuracy)
        
        if verbose :
            print("\n\n")
    
    fig, ax = plt.subplots()

    plt.xlim(0, epochs)
    plt.ylim(0, y_training[0])

    ax.plot(x_plot, y_training, label="Training")
    ax.plot(x_plot[1:], y_validation, label="Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    plt.legend()
    
    ax2 = ax.twinx()
    ax2.plot(x_plot[1:], y_validation_accuracy, label="Validation Accuracy", color="green")
    ax2.set_ylabel("Accuracy")

    plt.legend()

    plt.show()