from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str
    PROJECT_VERSION: str
    DATABASE_URL: str
    SENTIMENT_MODEL_PATH: str
    SENTIMENT_TOKENIZER_PATH: str
    NER_MODEL_PATH: str
    NER_TOKENIZER_PATH: str
    PII_MODEL_PATH: str
    PII_TOKENIZER_PATH: str
    POS_MODEL_PATH: str
    POS_TOKENIZER_PATH: str
    TOKEN_KEY: str

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()
