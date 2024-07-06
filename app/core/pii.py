import torch
from torch.nn.functional import softmax
from transformers import BertForTokenClassification
from transformers import BertTokenizer


class PIIModel:
    def __init__(self):
        self.model = BertForTokenClassification.from_pretrained(
            '/Users/abdukuddus/University of Greenwich/MSc Project/sentiment-analysis-app/pii-prototype/saved_model')
        self.tokenizer = BertTokenizer.from_pretrained(
            '/Users/abdukuddus/University of Greenwich/MSc Project/sentiment-analysis-app/pii-prototype/saved_model')

        self.tags_map = {'B-EMAIL': 0, 'B-NAME_STUDENT': 1, 'B-PHONE_NUM': 2, 'B-STREET_ADDRESS': 3,
                         'B-URL_PERSONAL': 4,
                         'B-USERNAME': 5, 'I-NAME_STUDENT': 6, 'I-PHONE_NUM': 7, 'I-STREET_ADDRESS': 8, 'O': 9}
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

        pii_list = [
            {"token": token, "label": label, "prob": round(prob * 100, 2)}
            for token, label, prob in zip(tokens, predicted_labels, predicted_probs)
            if token not in ['[CLS]', '[SEP]']
        ]

        return pii_list


pii_model = PIIModel()
