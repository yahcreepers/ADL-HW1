import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import csv

import torch

from dataset import SeqClsDataset
from model import LSTMModel, RNNModel, GRUModel, GRU_ATTModel
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

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
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

    model = GRU_ATTModel(300, hidden_size, n_layers, dropout, bid, 150)
    #model = LSTMModel(300, hidden_size, n_layers, dropout, bid, 150)
    model = model.cuda(c)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    idx2intent = ["" for i in range(150)]
    for i in intent2idx:
        idx2intent[intent2idx[i]] = i
    # load weights into model
    
    # TODO: predict dataset
    ans = ["cancel" for i in range(len(dataset))]
    for ind, datas in enumerate(test_loader):
        #print(ind, datas)
        bat = []
        for i in range(len(datas["text"])):
            text2vec = []
            l = 0
            for word in datas["text"][i].split(" "):
                if word in vocab.token2idx:
                    text2vec.append(embeddings[vocab.token2idx[word]])
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
        output = predictions.argmax(1)
        for i in range(len(output)):
            ans[eval(bat[i][-2].split("-")[1])] = idx2intent[output[i]]
    # TODO: write prediction to file (args.pred_file)
    header = ["id", "intent"]
    with open(args.pred_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i in range(len(ans)):
            temp = ["test-" + str(i), ans[i]]
            writer.writerow(temp)
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
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")
    parser.add_argument("--cuda", type=int, default=0)
    # data
    parser.add_argument("--max_len", type=int, default=128)

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
