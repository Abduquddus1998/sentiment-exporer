from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_db, get_current_user
from app.repository.user import create_user, get_user_by_email, get_user_by_id, get_users
from app.schemas.user import UserCreate, User

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/create", response_model=User)
def create_new_user(user: UserCreate, db: Session = Depends(get_db)):
    is_user_exist = get_user_by_email(db, email=user.email)

    if is_user_exist:
        raise HTTPException(status_code=400, detail={"error_message": "Email already registered"})

    return create_user(db, user)


@router.get("/all", response_model=List[User])
def get_all_users(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    users_list = get_users(db)

    if len(users_list) == 0:
        raise HTTPException(status_code=200, detail={"message": "Users not found"})

    return users_list


@router.get("/{user_id}", response_model=User)
def get_user(user_id: UUID, db: Session = Depends(get_db)):
    user = get_user_by_id(db, user_id=user_id)

    if user is None:
        raise HTTPException(status_code=400, detail={"error_message": "User not found"})

    return user

# @app.get("/users/me", response_model=schemas.UserInDB)
# def read_users_me(current_user: schemas.UserInDB = Depends(auth.get_current_user)):
#     return current_user
