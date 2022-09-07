import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import csv

import torch

from dataset import SeqClsDataset
from model import LSTMModel, RNNModel, GRUModel, SLOT_GRUModel
from torch.utils.data import Dataset,DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    for d in data:
        d["tokens"] = " ".join(d["tokens"])
        if "tags" in d:
            d["tags"] = " ".join(d["tags"])
    dataset = SeqClsDataset(data, vocab, tag2idx, args.max_len)
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    n_layers = args.num_layers
    dropout = args.dropout
    bid = args.bidirectional
    c = args.cuda
    
    # TODO: crecate DataLoader for test dataset
    test_loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=4
                        )
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    embeddings = embeddings.tolist()

    model = SLOT_GRUModel(300, hidden_size, n_layers, dropout, bid, 9)
    #model = LSTMModel(300, hidden_size, n_layers, dropout, bid, 150)
    model = model.cuda(c)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    idx2tag = ["" for i in range(150)]
    for i in tag2idx:
        idx2tag[tag2idx[i]] = i
    # load weights into model
    
    # TODO: predict dataset
    ans = [[] for i in range(len(dataset))]
    for ind, datas in enumerate(test_loader):
        #print(ind, datas)
        bat = []
        for i in range(len(datas["tokens"])):
            text2vec = []
            l = 0
            for word in datas["tokens"][i].split(" "):
                if word in vocab.token2idx:
                    text2vec.append(embeddings[vocab.token2idx[word]])
                    l += 1
                else:
                    text2vec.append([0 for i in range(300)])
                    l += 1
            #print(text, l)
            if(l != 0):
                text2vec.append(datas["id"][i])
                text2vec.append(l)
                bat.append(text2vec)
        bat = sorted(bat, key = lambda s: -s[-1])
        L = [i[-1] for i in bat]
        input = pad_sequence([torch.FloatTensor(i[:-2]) for i in bat], batch_first=True)
        
        input = input.cuda(c)
        
        predictions = model(input, L, c)
        for i in range(len(predictions)):
            for j in range(L[i]):
                ans[eval(bat[i][-2].split("-")[1])].append(idx2tag[predictions[i][j].argmax(0)])
    # TODO: write prediction to file (args.pred_file)
    header = ["id", "tags"]
    with open(args.pred_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i in range(len(ans)):
            temp = ["test-" + str(i), " ".join(ans[i])]
            writer.writerow(temp)
#    count = 0
#    for i in range(len(ans)):
#        if(len(ans[i]) != len(dataset[i]["tokens"].split(" "))):
#            print(ans[i])
#            print(dataset[i]["tokens"])
#            count += 1
#    print(count)
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--cuda", type=int, default=0)
    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

