from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.ner import ner_model
from app.core.pii import pii_model
from app.core.pos import pos_model
from app.dependencies import get_current_user, get_db
from app.repository.analysis import handle_analysis, get_sentiments_list, get_analysis_history, get_analysis_data
from app.schemas.analysis import AnalysisResponse, AnalysisBase, PIIResponse, NERResponse, POSResponse, \
    AnalysisResponseList, AnalysisHistoryResponse, AnalysisInfoResponse, AnalysisInfoParams
from app.schemas.user import User

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/sentiment", response_model=AnalysisResponse)
def create_analysis(data: AnalysisBase, current_user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    sentiment_data = handle_analysis(db=db, params=data)

    return {"success": True, "data": sentiment_data, "error": None}


@router.get("/sentiments-list", response_model=AnalysisResponseList)
def get_sentiment_analysis(current_user: User = Depends(get_current_user),
                           db: Session = Depends(get_db)):
    sentiment_list = get_sentiments_list(db=db)

    return {"success": True, "data": sentiment_list, "error": None}


@router.get("/history", response_model=AnalysisHistoryResponse)
def get_history(current_user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    history = get_analysis_history(db=db)

    return {"success": True, "data": history, "error": None}


@router.post("/analysis-info", response_model=AnalysisInfoResponse)
def get_analysis_info(params: AnalysisInfoParams, current_user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    analysis_info = get_analysis_data(db=db, corpus_id=params.corpus_id)

    return {"success": True, "data": analysis_info, "error": None}


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
