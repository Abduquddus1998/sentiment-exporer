from fastapi import APIRouter, Depends

from app.core.key_phrases import extract_key_phrases
from app.core.ner import ner_model
from app.core.pii import pii_model
from app.core.pos import pos_model
from app.core.sentiment import sentiment_model
from app.dependencies import get_current_user
from app.schemas.analysis import SentimentResponse, AnalysisBase, PIIResponse, NERResponse, POSResponse
from app.schemas.user import User

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/sentiment", response_model=SentimentResponse)
def sentiment_analysis(data: AnalysisBase, current_user: User = Depends(get_current_user)):
    sentiment, probability = sentiment_model.predict_sentiment(data.corpus)
    pii_labels = pii_model.predict_labels(data.corpus)
    ner_labels = ner_model.predict_labels(data.corpus)
    pos_labels = pos_model.predict_labels(data.corpus)
    key_phrases = extract_key_phrases(data.corpus)

    return {"sentiment": sentiment, "probability": probability, "corpus": data.corpus, "pii_labels": pii_labels,
            "ner_labels": ner_labels, "pos_labels": pos_labels, "key_phrases": key_phrases}


@router.post("/pii", response_model=PIIResponse)
def pii_analysis(data: AnalysisBase, current_user: User = Depends(get_current_user)):
    labels = pii_model.predict_labels(data.corpus)

    return {"corpus": data.corpus, "labels": labels}


@router.post("/ner", response_model=NERResponse)
def ner_analysis(data: AnalysisBase, current_user: User = Depends(get_current_user)):
    labels = ner_model.predict_labels(data.corpus)

    return {"corpus": data.corpus, "labels": labels}


@router.post("/pos", response_model=POSResponse)
def ner_analysis(data: AnalysisBase, current_user: User = Depends(get_current_user)):
    labels = pos_model.predict_labels(data.corpus)

    return {"corpus": data.corpus, "labels": labels}
