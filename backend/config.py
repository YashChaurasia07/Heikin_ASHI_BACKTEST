from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "heikin_ashi_db"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # CORS
    CORS_ORIGINS: str = "*"
    
    # TradingView
    TV_USERNAME: str = ""
    TV_PASSWORD: str = ""
    
    # Data
    DEFAULT_START_DATE: str = "2022-01-01"
    DEFAULT_INITIAL_CAPITAL: float = 50000
    DATA_UPDATE_HOURS: int = 24  # Hours before data is considered stale
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def cors_origins_list(self) -> List[str]:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


settings = Settings()
