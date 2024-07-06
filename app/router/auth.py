from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import verify_password
from app.dependencies import get_db
from app.repository.auth import create_access_token
from app.repository.user import create_user, get_user_by_email
from app.schemas.user import UserCreate, AuthResponse, User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def create_new_user(user: UserCreate, db: Session = Depends(get_db)):
    is_user_exist = get_user_by_email(db, email=user.email)

    if is_user_exist:
        return {"success": False, "data": None, "error": {"message": "Email already registered"}}

    new_user = create_user(db, user=user)
    access_token = create_access_token(data={"sub": new_user.email})
    user_data = User.from_orm(new_user)

    data = {
        "access_token": access_token,
        "token_type": "bearer",
        **user_data.dict()
    }

    return {"success": True, "data": data, "error": None}


@router.post("/login", response_model=AuthResponse)
def login(user: UserCreate, db: Session = Depends(get_db)):
    is_user_exist = get_user_by_email(db, email=user.email)

    if not is_user_exist:
        return {"success": False, "data": None, "error": {"message": "User does not exist, please sign up first"}}
        # raise HTTPException(status_code=401, detail={"error_message": "User does not exist, please sign up first"})

    is_password_correct = verify_password(user.password, is_user_exist.hashed_password)

    if not is_password_correct:
        return {"success": False, "data": None, "error": {"message": "Incorrect username or password"}}
        # raise HTTPException(status_code=401, detail={"error_message": "Incorrect username or password"})

    access_token = create_access_token(data={"sub": is_user_exist.email})
    user_data = User.from_orm(is_user_exist)

    data = {
        "access_token": access_token,
        "token_type": "bearer",
        **user_data.dict()
    }

    return {"success": True, "data": data, "error": None}
