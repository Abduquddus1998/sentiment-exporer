from typing import List

from pydantic import BaseModel


class AnalysisBase(BaseModel):
    corpus: str


class LabelAndProb(BaseModel):
    token: str
    label: str
    prob: float


class PIIResponse(AnalysisBase):
    labels: List[LabelAndProb]


class NERResponse(AnalysisBase):
    labels: List[LabelAndProb]


class POSResponse(AnalysisBase):
    labels: List[LabelAndProb]


class KeyPhrases(BaseModel):
    score: float
    phrase: str


class SentimentResponse(AnalysisBase):
    sentiment: str
    probability: float
    pii_labels: List[LabelAndProb]
    ner_labels: List[LabelAndProb]
    pos_labels: List[LabelAndProb]
    key_phrases: List[KeyPhrases]
