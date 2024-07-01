from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import verify_password
from app.dependencies import get_db
from app.repository.auth import create_access_token
from app.repository.user import create_user, get_user_by_email
from app.schemas.user import UserCreate, NewUser, User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=NewUser)
def create_new_user(user: UserCreate, db: Session = Depends(get_db)):
    is_user_exist = get_user_by_email(db, email=user.email)

    if is_user_exist:
        raise HTTPException(status_code=400, detail={"error_message": "Email already registered"})

    new_user = create_user(db, user=user)
    access_token = create_access_token(data={"sub": new_user.email})
    user_data = User.from_orm(new_user)

    return {"access_token": access_token, "token_type": "bearer", **user_data.dict()}


@router.post("/login", response_model=NewUser)
def login(user: UserCreate, db: Session = Depends(get_db)):
    is_user_exist = get_user_by_email(db, email=user.email)

    if not is_user_exist:
        raise HTTPException(status_code=401, detail={"error_message": "User does not exist, please sign up first"})

    is_password_correct = verify_password(user.password, is_user_exist.hashed_password)

    if not is_password_correct:
        raise HTTPException(status_code=401, detail={"error_message": "Incorrect username or password"})

    access_token = create_access_token(data={"sub": is_user_exist.email})
    user_data = User.from_orm(is_user_exist)

    return {"access_token": access_token, "token_type": "bearer", **user_data.dict(), }
