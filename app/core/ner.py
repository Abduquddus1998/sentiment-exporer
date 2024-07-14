import torch
from torch.nn.functional import softmax
from transformers import BertForTokenClassification
from transformers import BertTokenizer

from app.core.config import settings


class NerModel:
    def __init__(self):
        self.model = BertForTokenClassification.from_pretrained(
            settings.NER_MODEL_PATH)
        self.tokenizer = BertTokenizer.from_pretrained(
            settings.NER_TOKENIZER_PATH)

        self.tags_map = {'O': 0, 'I-ORG': 1, 'I-PER': 2, 'B-GPE': 3, 'I-GPE': 4, 'B-PER': 5, 'I-ART': 6, 'I-TIM': 7,
                         'B-GEO': 8,
                         'I-GEO': 9, 'B-NAT': 10, 'B-TIM': 11, 'B-EVE': 12, 'I-EVE': 13, 'I-NAT': 14, 'B-ART': 15,
                         'B-ORG': 16}

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

        ner_list = []
        current_word = ""
        current_label = None
        current_prob = None

        for token, label, prob in zip(tokens, predicted_labels, predicted_probs):
            if token in ['[CLS]', '[SEP]']:
                continue
            if token.startswith("##"):
                current_word += token[2:]
            else:
                if current_word:
                    ner_list.append(
                        {"token": current_word, "label": current_label, "prob": round(current_prob * 100, 2)})
                current_word = token
                current_label = label
                current_prob = prob
        if current_word:
            ner_list.append({"token": current_word, "label": current_label, "prob": round(current_prob * 100, 2)})

        return ner_list


ner_model = NerModel()
