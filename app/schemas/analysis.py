from typing import List
from uuid import UUID

from pydantic import BaseModel


class AnalysisBase(BaseModel):
    corpus: str


class LabelAndProb(BaseModel):
    id: UUID
    corpus_id: UUID
    token: str
    label: str
    prob: float

    class Config:
        orm_mode = True
        from_attributes = True


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
    corpus_id: UUID
    corpus: str
    sentiment: str
    prob: float
    pii_labels: List[LabelAndProb]
    ner_labels: List[LabelAndProb]
    pos_labels: List[LabelAndProb]
    key_phrases: List[KeyPhrases]

    class Config:
        orm_mode = True
        from_attributes = True


class AnalysisResponse(BaseModel):
    success: bool
    data: SentimentResponse | None
    error: None | dict


class AnalysisResponseList(BaseModel):
    success: bool
    data: List[SentimentResponse] | None
    error: None | dict


class AnalysisHistory(AnalysisBase):
    corpus_id: UUID


class AnalysisHistoryResponse(BaseModel):
    success: bool
    data: List[AnalysisHistory] | None
    error: None | dict


class AnalysisInfoParams(BaseModel):
    corpus_id: UUID


class AnalysisInfoResponse(BaseModel):
    success: bool
    data: SentimentResponse | None
    error: None | dict
