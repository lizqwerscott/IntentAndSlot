from transformers import BertTokenizer
import transformers
import torch
import numpy as np
import os
from seqeval.metrics.sequence_labeling import get_entities

from model import BertForIntentClassificationAndSlotFilling


class Args:
    seq_labels_path = "./data/intents.txt"
    token_labels_path = "./data/slots.txt"
    bert_dir = "./bert/"
    load_dir = "./data/model.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    id2seqlabel = {}
    with open(seq_labels_path, "r") as fp:
        seq_labels = fp.read().split("\n")
        for i, label in enumerate(seq_labels):
            id2seqlabel[i] = label

    with open(token_labels_path, "r") as fp:
        token_labels = fp.read().split("\n")

    tmp = ["O"]
    for label in token_labels:
        B_label = "B-" + label
        I_label = "I-" + label
        tmp.append(B_label)
        tmp.append(I_label)

    id2nerlabel = {}
    for i, label in enumerate(tmp):
        id2nerlabel[i] = label

    hidden_size = 768
    seq_num_labels = len(seq_labels)
    token_num_labels = len(tmp)
    max_len = 32
    hidden_dropout_prob = 0.1


class Predicter:
    def __init__(self, config):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.model = BertForIntentClassificationAndSlotFilling(config)
        self.model.load_state_dict(torch.load(config.load_dir))
        self.model.to(config.device)

        self.config = config

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            tmp_text = [i for i in text]
            inputs = self.tokenizer.encode_plus(
                text=tmp_text,
                max_length=self.config.max_len,
                padding="max_length",
                truncation="only_first",
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors="pt",
            )
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.config.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            token_type_ids = inputs["token_type_ids"]
            seq_output, token_output = self.model(
                input_ids,
                attention_mask,
                token_type_ids,
            )
            seq_output = seq_output.detach().cpu().numpy()
            token_output = token_output.detach().cpu().numpy()
            seq_output = np.argmax(seq_output, -1)
            token_output = np.argmax(token_output, -1)
            seq_output = seq_output[0]
            token_output = token_output[0][1 : len(text) - 1]
            token_output = [self.config.id2nerlabel[i] for i in token_output]

            intent = self.config.id2seqlabel[seq_output]
            slots = [
                (i[0], text[i[1] : i[2] + 1], i[1], i[2])
                for i in get_entities(token_output)
            ]
            slots_r = []
            for item in slots:
                slot = { "key": item[0], "value": item[1] }
                slots_r.append(slot)

            print("意图：", intent)
            print(
                "槽位：",
                str(slots),
            )
            return { "intent": intent, "slots": slots_r }


if __name__ == "__main__":
    args = Args()
    predicter = Predicter(args)

    with open("./data/test2.json", "r") as fp:
        pred_data = eval(fp.read())
        for i, p_data in enumerate(pred_data):
            text = p_data["text"]
            print("=================================")
            print(text)
            predicter.predict(text)
            print("=================================")
            if i == 10:
                break
