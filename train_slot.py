import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange
from torch.utils.data import Dataset,DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from model import LSTMModel, SLOT_GRUModel, weight_init
import numpy as np

from dataset import SeqClsDataset
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
ACC = 0

def run(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    for let in SPLITS:
        for d in data[let]:
            d["tokens"] = " ".join(d["tokens"])
            d["tags"] = " ".join(d["tags"])
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    embeddings = embeddings.tolist()
    
    # TODO: crecate DataLoader for train / dev datasets
    #print(datasets["train"].data)
    print("SSS", args.batch_size, args.hidden_size, args.dropout)
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    n_layers = args.num_layers
    dropout = args.dropout
    bid = args.bidirectional
    lr = args.lr
    c = args.cuda
    
    train_loader = DataLoader(dataset=datasets["train"],
                        batch_size=batch_size,
                        pin_memory=True,
                        shuffle=True,
                        num_workers=4
                        )
    eval_loader = DataLoader(dataset=datasets["eval"],
                        batch_size=batch_size,
                        pin_memory=True,
                        shuffle=True,
                        num_workers=4
                        )
    
    #print(datasets["train"].vocab.token2idx)
    #print(len(train_loader), len(datasets["train"]))
    def train(model, optimizer, criterion):
        e_l = 0
        e_a = 0
        t_a = 0
        ln = 0
        cor = 0
        tol = 0
        model.train()
        for ind, datas in enumerate(train_loader):
            bat = []
            ID = 0
            for text in datas["tokens"]:
                text2vec = []
                l = 0
                for word in text.split(" "):
                    if word in vocab.token2idx:
                        text2vec.append(embeddings[vocab.token2idx[word]])
                        l += 1
                text2vec.append(ID)
                text2vec.append(l)
                bat.append(text2vec)
                ID += 1
            bat = sorted(bat, key = lambda s: -s[-1])
            L = [i[-1] for i in bat]
            y = []
            labels = []
            for i in bat:
                a = []
                b = []
                for j in datas["tags"][i[-2]].split(" "):
                    y_i = [0 for i in range(9)]
                    y_i[tag2idx[j]] = 1
                    a.append(y_i)
                    b.append(tag2idx[j])
                y.append(a)
                labels.append(b)
            
            bat = pad_sequence([torch.FloatTensor(i[:-2]) for i in bat], batch_first=True)
            y = pad_sequence([torch.FloatTensor(i) for i in y], batch_first=True)
            
            bat = bat.cuda(c)
            y = y.cuda(c)
            
            #bat.to(device)
            #label = [0 for i in range(150)]
            #label[intent2idx[datas]] = 1
            optimizer.zero_grad()
            predictions = model(bat, L, c)
            #e_a += (predicted == labels).sum()
            for i in range(len(predictions)):
                flag = 0
                for j in range(L[i]):
                    if(predictions[i][j].argmax(0) != labels[i][j]):
                        flag = 1
                    else:
                        t_a += 1
                    ln += 1
                e_a += not flag
            #e_a += (predictions.argmax(1) == labels)
            #print(predictions, predictions.size())
            #print(y)
            loss = criterion(predictions, y)
            
            loss.backward()
            optimizer.step()
            e_l += loss.item()
        print(e_l/len(train_loader), e_a/len(datasets["train"]), e_a, t_a/ln)
        return e_l/len(train_loader), e_a/len(datasets["train"])
    def eval(model, criterion):
        e_l = 0
        e_a = 0
        t_a = 0
        ln = 0
        model.eval()
        with torch.no_grad():
            for ind, datas in enumerate(eval_loader):
                bat = []
                ID = 0
                for text in datas["tokens"]:
                    text2vec = []
                    l = 0
                    for word in text.split(" "):
                        if word in vocab.token2idx:
                            text2vec.append(embeddings[vocab.token2idx[word]])
                            l += 1
                    text2vec.append(ID)
                    text2vec.append(l)
                    bat.append(text2vec)
                    ID += 1
                bat = sorted(bat, key = lambda s: -s[-1])
                L = [i[-1] for i in bat]
                y = []
                labels = []
                for i in bat:
                    a = []
                    b = []
                    for j in datas["tags"][i[-2]].split(" "):
                        y_i = [0 for i in range(9)]
                        y_i[tag2idx[j]] = 1
                        a.append(y_i)
                        b.append(tag2idx[j])
                    y.append(a)
                    labels.append(b)
                
                bat = pad_sequence([torch.FloatTensor(i[:-2]) for i in bat], batch_first=True)
                y = pad_sequence([torch.FloatTensor(i) for i in y], batch_first=True)
                
                bat = bat.cuda(c)
                y = y.cuda(c)
                
                #bat.to(device)
                #label = [0 for i in range(150)]
                #label[intent2idx[datas]] = 1
                predictions = model(bat, L, c)
                #e_a += (predicted == labels).sum()
                for i in range(len(predictions)):
                    flag = 0
                    for j in range(L[i]):
                        if(predictions[i][j].argmax(0) != labels[i][j]):
                            flag = 1
                        else:
                            t_a += 1
                        ln += 1
                    e_a += not flag
                #e_a += (predictions.argmax(1) == labels)
                #print(predictions, predictions.size())
                #print(y)
                loss = criterion(predictions, y)
                e_l += loss.item()
        print(e_l/len(eval_loader), e_a/len(datasets["eval"]), e_a, t_a/ln)
        return e_l/len(eval_loader), e_a/len(datasets["eval"])
    # TODO: init model and move model to target device(cpu / gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device')
    #model = SLOT_GRUModel(300, hidden_size, n_layers, dropout, bid, 9)
    model = SLOT_GRUModel(300, hidden_size, n_layers, dropout, bid, 9)
    #T = nn.RNN(300, hidden_size, n_layers, dropout=dropout, bidirectional=bid, batch_first=False, nonlinearity='relu')
    #print(vocab.token2idx)
    
    # TODO: init optimizer
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model = model.cuda(c)
    criterion = criterion.cuda(c)
    model.apply(weight_init)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    X = 0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        t_loss, t_acc = train(model, optimizer, criterion)
        e_loss, e_acc = eval(model, criterion)
        if(e_acc > X):
            X = e_acc
            T_L = t_loss
            T_A = t_acc
            E_L = e_loss
            E_A = e_acc
            if(X > ACC):
                torch.save(model.state_dict(), str(args.ckpt_dir) + "/" + 'best.pt')
    return X, T_L, T_A, E_L, E_A
    # TODO: Inference on test set
def main(args):
    global ACC
    if(args.FindParam):
        #batch_sizes = [4, 8, 16, 32]
        batch_sizes = [128]
        hidden_sizes = [512, 1024]
        dropouts = [0.1, 0.2, 0.4]
        #epoches = [20, 40, 60, 80]
        epoches = [100]
        best_B = 128
        best_H = 256
        best_D = 0.1
        ANS = []
        for i in range(len(batch_sizes)):
            for j in hidden_sizes:
                for k in dropouts:
                    args.batch_size = batch_sizes[i]
                    args.hidden_size = j
                    args.dropout = k
                    args.num_epoch = epoches[i]
                    ANS.append((j, k, run(args)))
                    acc = ANS[-1][0]
                    if(acc > ACC):
                        print(batch_sizes[i], j, k)
                        best_B = batch_sizes[i]
                        best_H = j
                        best_D = k
                        ACC = acc
        print("best = ", best_B, best_H, best_D, ACC)
        for ans in ANS:
            print("hidden size = ", ans[0], "dropouts = ", ans[1])
            ans = ans[2]
            print("best = ", ans[0], "Train_Loss = ", ans[1], "Train_Acc = ", ans[2], "Test_Loss = ", ans[3], "Test_Acc = ", ans[4])
    else:
        ans = run(args)
        print("best = ", ans[0], "Train_Loss = ", ans[1], "Train_Acc = ", ans[2], "Test_Loss", ans[3], "Test_Acc", ans[4], sep = "")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--cuda", type=int, default=0)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--FindParam", action="store_true")

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

