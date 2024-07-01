from uuid import UUID

from pydantic import BaseModel


class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: UUID
    is_active: bool

    class Config:
        orm_mode = True
        from_attributes = True


class NewUser(UserBase):
    id: UUID
    is_active: bool
    access_token: str
    token_type: str

    class Config:
        orm_mode = True
        from_attributes = True
