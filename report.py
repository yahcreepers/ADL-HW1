import json
from argparse import ArgumentParser

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


def main(args):
    with open(args.eval_json) as f:
        eval_data = json.load(f)
        truth = [e["tags"] for e in eval_data]
    #print(truth)
    with open(args.predict_csv) as f:
        predictions = []
        for line in f.readlines()[1:]:
            #print(line.split(",")[1].split())
            predictions.append(line.split(",")[1].split())
#    for i in range(len(truth)):
#        print(truth[i], len(truth[i]))
#        print(predictions[i], len(truth[i]))
            
    print(classification_report(truth, predictions, mode="strict", scheme=IOB2))


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--eval_json", default="data/slot/eval.json")
    parser.add_argument("--predict_csv", default="pred.slot.eval.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
