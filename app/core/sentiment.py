import numpy as np
import torch
from cleantext import clean
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification

from app.core.config import settings


class SentimentModel:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained(
            settings.SENTIMENT_MODEL_PATH)
        self.tokenizer = BertTokenizer.from_pretrained(
            settings.SENTIMENT_TOKENIZER_PATH)
        self.label_encoder = LabelEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def text_cleaner(self, text: str):
        return clean(text,
                     fix_unicode=True,
                     to_ascii=True,
                     lower=True,
                     no_line_breaks=True,
                     no_urls=True,
                     replace_with_url="",
                     no_emails=True,
                     replace_with_email="",
                     no_phone_numbers=True,
                     no_punct=False,
                     replace_with_phone_number="",
                     lang="en")

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits))

        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    def predict_sentiment(self, sentence: str):
        self.label_encoder.fit(['negative', 'neutral', 'positive'])

        cleaned_sentence = self.text_cleaner(sentence)
        encoding = self.tokenizer(cleaned_sentence, return_tensors='pt', truncation=True, padding=True)
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        logits = outputs.logits[0].cpu().tolist()
        probs = self.softmax(logits)
        pred_class = np.argmax(probs)
        pred_label = self.label_encoder.inverse_transform([pred_class])[0]
        pred_percentage = probs[pred_class]
        rounded_pred_percentage = round(pred_percentage * 100, 2)

        return pred_label, rounded_pred_percentage


sentiment_model = SentimentModel()
