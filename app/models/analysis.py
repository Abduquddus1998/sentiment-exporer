import uuid

from sqlalchemy import Column, String, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.database import Base


class Sentiment(Base):
    __tablename__ = 'sentiments'
    corpus_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, index=True)
    corpus = Column(String)
    prob = Column(Float)
    sentiment = Column(String)
    pii_labels = relationship("PII", back_populates="sentiment")
    ner_labels = relationship("NER", back_populates="sentiment")
    pos_labels = relationship("POS", back_populates="sentiment")
    key_phrases = relationship("KeyPhrases", back_populates="sentiment")


class PII(Base):
    __tablename__ = 'pii_labels'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, index=True)
    corpus_id = Column(UUID(as_uuid=True), ForeignKey("sentiments.corpus_id"), nullable=False)
    token = Column(String, nullable=False)
    label = Column(String, nullable=False)
    prob = Column(Float, nullable=False)
    sentiment = relationship("Sentiment", back_populates="pii_labels")


class NER(Base):
    __tablename__ = 'ner_labels'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, index=True)
    corpus_id = Column(UUID(as_uuid=True), ForeignKey("sentiments.corpus_id"), nullable=False)
    token = Column(String, nullable=False)
    label = Column(String, nullable=False)
    prob = Column(Float)
    sentiment = relationship("Sentiment", back_populates="ner_labels")


class POS(Base):
    __tablename__ = 'pos_labels'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, index=True)
    corpus_id = Column(UUID(as_uuid=True), ForeignKey("sentiments.corpus_id"), nullable=False)
    token = Column(String, nullable=False)
    label = Column(String, nullable=False)
    prob = Column(Float)
    sentiment = relationship("Sentiment", back_populates="pos_labels")


class KeyPhrases(Base):
    __tablename__ = 'key_phrases'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, index=True)
    corpus_id = Column(UUID(as_uuid=True), ForeignKey("sentiments.corpus_id"), nullable=False)
    phrase = Column(String, nullable=False)
    score = Column(Float)
    sentiment = relationship("Sentiment", back_populates="key_phrases")
