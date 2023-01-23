import torch
import random
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pad = 0
sos = 1
eos = 2

def train_flip(N, N_Val, size = 9):
    x = torch.randint(3, 9, (N + N_Val, size)).to(device) + pad
    x[:, 0] = sos
    x[:, -1] = eos

    trg = torch.flip(x, [1])
    trg[:, 0] = sos
    trg[:, -1] = eos

    cat_trg = torch.zeros((N + N_Val, size, 10)).to(device)
    for i, t in enumerate(trg):
        for j, word in enumerate(t):
            cat_trg[i, j, word] = 1

    return x, trg, cat_trg

def train_fixed_freq_v1(N, N_Val, f = 20):
    sentences = torch.randint(3, 100, (N + N_Val, f)).to(device)
    x = torch.zeros((N + N_Val, 3 * f + 2)).to(device).int() + pad
    x[:, 0] = sos
    x[:, 1:f + 1] = sentences
    x[:, f + 1:2 * f + 1] = sentences
    x[:, 2 * f + 1:3 * f + 1] = sentences
    x[:, -1] = eos

    trg = x

    cat_trg = torch.zeros((N + N_Val, 3 * f + 2, 100)).to(device)
    for i, sentence in enumerate(trg):
        for j, word in enumerate(sentence):
            cat_trg[i, j, word] = 1

    return x, trg, cat_trg

def train_fixed_freq_v2(N, N_Val, f = 20):
    sentences = torch.randint(3, 9, (N + N_Val, f)).to(device)
    x = torch.zeros((N + N_Val, 2 * f + 2)).to(device).int() + pad
    x[:, 0] = sos
    x[:, 1:f + 1] = sentences
    x[:, f + 1:2 * f + 1] = sentences
    x[:, -1] = eos

    trg = torch.zeros((N + N_Val, 3 * f + 2)).to(device).int() + pad
    trg[:, 0] = sos
    trg[:, 1:f + 1] = sentences
    trg[:, f + 1:2 * f + 1] = sentences
    trg[:, 2 * f + 1:3 * f + 1] = sentences
    trg[:, -1] = eos

    cat_trg = torch.zeros((N + N_Val, 3 * f + 2, 10)).to(device)
    for i, sentence in enumerate(trg):
        for j, word in enumerate(sentence):
            cat_trg[i, j, word] = 1

    return x, trg, cat_trg

def train_fixed_freq_v3(N, N_Val, f = 20):
    sentences = torch.randint(3, 9, (N + N_Val, f)).to(device)
    x = torch.zeros((N + N_Val, 3 * f + 2)).to(device).int() + pad
    x[:, 0] = sos
    x[:, 1:f + 1] = sentences
    x[:, f + 1:2 * f + 1] = sentences
    x[:, 2 * f + 1:3 * f + 1] = sentences
    x[:, -1] = eos

    trg = torch.zeros((N + N_Val, f + 2)).to(device).int() + pad
    trg[:, 0] = sos
    trg[:, 1:f + 1] = sentences
    trg[:, -1] = eos

    cat_trg = torch.zeros((N + N_Val, f + 2, 10)).to(device)
    for i, sentence in enumerate(trg):
        for j, word in enumerate(sentence):
            cat_trg[i, j, word] = 1

    return x, trg, cat_trg

def train_random_freq(N, N_Val, fmax = 20):
    #la taille des séquences sera variable
    #PB : il va vouloir recopier le début, parce que si à aucun moment il recopie ce qu'il faut, il pourra jamais utiliser ses tokens déjà créés
    #       donc il fait juste une identité
    #solution : séquence de taille fixée, fréquence non fixée, cf fonction suivante
    x = torch.zeros((N + N_Val, 3 * fmax + 2)).to(device).int() + pad
    x[:, 0] = sos

    trg = torch.zeros((N + N_Val, fmax + 2)).to(device).int() + pad
    trg[:, 0] = sos

    for i in range(N + N_Val):
        f = random.randint(fmax // 2, fmax)
        sentence = torch.randint(3, 9, (1, f)).to(device)
        x[i, 1:f + 1] = sentence
        x[i, f + 1:2 * f + 1] = sentence
        x[i, 2 * f + 1:3 * f + 1] = sentence
        x[i, 3 * f + 1] = eos

        trg[i, 1:f + 1] = sentence
        trg[:, f + 1] = eos

    cat_trg = torch.zeros((N + N_Val, fmax + 2, 10)).to(device)
    for i, sentence in enumerate(trg):
        for j, word in enumerate(sentence):
            cat_trg[i, j, word] = 1

    return x, trg, cat_trg

def train_radom_pattern(N, N_Val, size = 15, nb_concat = 1):
    #phrase random, dans laquelle on met un pattern aléatoirement, puis on donne du bruit puis le début du pattern et on demande la fin
    inter_max = 20 - size

    x = torch.randint(3, 100, (N + N_Val, nb_concat * 3 * 20 + 2)).to(device) + pad
    x[:, 0] = sos

    trg = torch.randint(3, 100, (N + N_Val, nb_concat * 2 * size + 2)).to(device) + pad
    trg[:, 0] = sos

    for i in range(N + N_Val):
        for k in range(nb_concat):
            inter_size = random.randint(0, inter_max)
            x[i, k * 60 + size + inter_size + 1:k * 60 + 2 * size + inter_size + 1] = x[i, k * 60 + 1:k * 60 + size + 1]
            x[i, k * 60 + 2 * size + 2 * inter_size + 1:k * 60 + 3 * size + 2 * inter_size + 1] = x[i, k * 60 + 1:k * 60 + size + 1]

            start_trg = random.randint(1, size)
            trg[i, k * 2 * size + start_trg:k * 2 * size + start_trg + size] = x[i, k * 60 + 1:k * 60 + size + 1]

    x[:, -1] = eos
    trg[:, -1] = eos

    cat_trg = torch.zeros((N + N_Val, nb_concat * 2 * size + 2, 100)).to(device)
    for i, sentence in enumerate(trg):
        if i % (len(trg) // 100) == 0 :
            print(str(i * 100 // len(trg)), "%")
        for j, word in enumerate(sentence):
            cat_trg[i, j, word] = 1

    return x, trg, cat_trg

def train_random_flip(N, N_Val):
    pass