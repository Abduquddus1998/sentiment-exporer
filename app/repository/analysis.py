from typing import List, Type
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.key_phrases import extract_key_phrases
from app.core.ner import ner_model
from app.core.pii import pii_model
from app.core.pos import pos_model
from app.core.sentiment import sentiment_model
from app.models.analysis import Sentiment, PII, NER, POS, KeyPhrases
from app.schemas.analysis import AnalysisBase


def handle_analysis(db: Session, params: AnalysisBase) -> Sentiment:
    sentiment_record = create_sentiment(db=db, corpus=params.corpus)

    create_pii(db=db, sentiment=sentiment_record)
    create_ner(db=db, sentiment=sentiment_record)
    create_pos(db=db, sentiment=sentiment_record)
    create_key_phrases(db=db, sentiment=sentiment_record)

    return sentiment_record


def create_sentiment(db: Session, corpus: str) -> Sentiment:
    sentiment, probability = sentiment_model.predict_sentiment(corpus)

    sentiment_record = Sentiment(corpus=corpus, sentiment=str(sentiment), prob=float(probability))
    db.add(sentiment_record)
    db.commit()
    db.refresh(sentiment_record)

    return sentiment_record


def create_pii(db: Session, sentiment: Sentiment) -> None:
    pii_labels = pii_model.predict_labels(sentiment.corpus)

    for pii in pii_labels:
        pii_record = PII(
            corpus_id=sentiment.corpus_id,
            token=pii["token"],
            label=pii["label"],
            prob=pii["prob"]
        )
        db.add(pii_record)

    db.commit()
    db.refresh(sentiment)


def create_ner(db: Session, sentiment: Sentiment) -> None:
    ner_labels = ner_model.predict_labels(sentiment.corpus)

    for ner in ner_labels:
        ner_record = NER(
            corpus_id=sentiment.corpus_id,
            token=ner["token"],
            label=ner["label"],
            prob=ner["prob"]
        )
        db.add(ner_record)

    db.commit()
    db.refresh(sentiment)


def create_pos(db: Session, sentiment: Sentiment) -> None:
    pos_labels = pos_model.predict_labels(sentiment.corpus)

    for pos in pos_labels:
        pos_record = POS(
            corpus_id=sentiment.corpus_id,
            token=pos["token"],
            label=pos["label"],
            prob=pos["prob"]
        )
        db.add(pos_record)

    db.commit()
    db.refresh(sentiment)


def create_key_phrases(db: Session, sentiment: Sentiment) -> None:
    key_phrases = extract_key_phrases(sentiment.corpus)

    for phrases in key_phrases:
        key_phrases_record = KeyPhrases(
            corpus_id=sentiment.corpus_id,
            score=phrases["score"],
            phrase=phrases["phrase"]
        )
        db.add(key_phrases_record)

    db.commit()
    db.refresh(sentiment)


def get_analysis_history(db: Session) -> List[Sentiment]:
    sentiments = db.query(Sentiment).all()

    sentiment_list = [
        Sentiment(
            corpus_id=sentiment.corpus_id,
            corpus=sentiment.corpus
        ) for sentiment in sentiments
    ]

    return sentiment_list


def get_analysis_data(db: Session, corpus_id: UUID) -> Sentiment:
    sentiment = db.query(Sentiment).filter(Sentiment.corpus_id == corpus_id).first()

    sentiment_data = Sentiment(
        corpus_id=sentiment.corpus_id,
        corpus=sentiment.corpus,
        prob=sentiment.prob,
        sentiment=sentiment.sentiment,
        pii_labels=get_pii_labels(sentiment),
        pos_labels=get_pos_labels(sentiment),
        ner_labels=get_ner_labels(sentiment),
        key_phrases=get_key_phrases(sentiment)

    )

    return sentiment_data


def get_sentiments_list(db: Session) -> List[Sentiment]:
    sentiments = db.query(Sentiment).all()

    sentiment_list = [
        Sentiment(
            corpus_id=sentiment.corpus_id,
            corpus=sentiment.corpus,
            prob=sentiment.prob,
            sentiment=sentiment.sentiment,
            pii_labels=get_pii_labels(sentiment),
            pos_labels=get_pos_labels(sentiment),
            ner_labels=get_ner_labels(sentiment),
            key_phrases=get_key_phrases(sentiment)

        ) for sentiment in sentiments
    ]

    return sentiment_list


def get_pii_labels(sentiment: Type[Sentiment]) -> List[PII]:
    return [
        PII(
            id=pii.id,
            corpus_id=pii.corpus_id,
            token=pii.token,
            label=pii.label,
            prob=pii.prob
        ) for pii in sentiment.pii_labels
    ]


def get_ner_labels(sentiment: Type[Sentiment]) -> List[NER]:
    return [
        NER(
            id=ner.id,
            corpus_id=ner.corpus_id,
            token=ner.token,
            label=ner.label,
            prob=ner.prob
        ) for ner in sentiment.ner_labels
    ]


def get_pos_labels(sentiment: Type[Sentiment]) -> List[POS]:
    return [
        POS(
            id=pos.id,
            corpus_id=pos.corpus_id,
            token=pos.token,
            label=pos.label,
            prob=pos.prob
        ) for pos in sentiment.pos_labels
    ]


def get_key_phrases(sentiment: Type[Sentiment]) -> List[KeyPhrases]:
    return [
        KeyPhrases(
            id=key_phrases.id,
            corpus_id=key_phrases.corpus_id,
            score=key_phrases.score,
            phrase=key_phrases.phrase
        ) for key_phrases in sentiment.key_phrases
    ]
