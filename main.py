from fastapi import FastAPI

from app.db.database import Base, engine
from app.router import users, auth

app = FastAPI()

app.include_router(users.router)
app.include_router(auth.router)

Base.metadata.create_all(bind=engine)
