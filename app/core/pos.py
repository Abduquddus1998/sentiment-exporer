import torch
from torch.nn.functional import softmax
from transformers import BertForTokenClassification
from transformers import BertTokenizer


class POSModel:
    def __init__(self):
        self.model = BertForTokenClassification.from_pretrained(
            '/Users/abdukuddus/University of Greenwich/MSc Project/sentiment-analysis-app/pos-prototype/saved_model')
        self.tokenizer = BertTokenizer.from_pretrained(
            '/Users/abdukuddus/University of Greenwich/MSc Project/sentiment-analysis-app/ner-prototype/saved_model')

        self.tags_map = {'EX': 0, 'VBD': 1, 'JJR': 2, 'PRP': 3, 'JJS': 4, 'SYM': 5, 'VBP': 6, ':': 7, 'VBG': 8,
                         'NNP': 9,
                         'VBZ': 10, '(': 11, 'FW': 12, ')': 13, 'MD': 14, "''": 15, 'VBN': 16, '$': 17, 'RP': 18,
                         'LS': 19,
                         'IN': 20, '"': 21, 'JJ': 22, 'RBR': 23, 'UH': 24, 'TO': 25, 'POS': 26, 'WP': 27, 'NNS': 28,
                         'VB': 29, 'NNPS': 30, 'WP$': 31, 'NN|SYM': 32, 'RBS': 33, 'NN': 34, 'CD': 35, 'WRB': 36,
                         'DT': 37,
                         'CC': 38, 'PRP$': 39, 'PDT': 40, 'WDT': 41, ',': 42, 'RB': 43, '.': 44, 'O': 45}

        self.reverse_tags_map = {v: k for k, v in self.tags_map.items()}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict_labels(self, text: str):
        tokenized_input = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512,
                                         is_split_into_words=False)

        tokenized_input = {key: val.to(self.device) for key, val in tokenized_input.items()}

        with torch.no_grad():
            output = self.model(**tokenized_input)

        probabilities = softmax(output.logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])
        predicted_labels = [self.reverse_tags_map[pred.item()] for pred in predictions[0]]
        predicted_probs = [prob[pred].item() for prob, pred in zip(probabilities[0], predictions[0])]

        pos_list = [
            {"token": token, "label": label, "prob": round(prob * 100, 2)}
            for token, label, prob in zip(tokens, predicted_labels, predicted_probs)
            if token not in ['[CLS]', '[SEP]']
        ]

        return pos_list


pos_model = POSModel()
