from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from database import connect_to_mongo, close_mongo_connection
from routes import data, symbols, backtest, advanced_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown"""
    # Startup
    logger.info("Starting Heikin Ashi Backtest Server...")
    await connect_to_mongo()
    logger.info("Server started successfully")
    yield
    # Shutdown
    logger.info("Shutting down server...")
    await close_mongo_connection()
    logger.info("Server stopped")


# Create FastAPI app
app = FastAPI(
    title="Heikin Ashi Backtest Server",
    description="Professional backtesting API with MongoDB storage",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware - Allow all origins for Vercel
# In production, configure CORS_ORIGINS env variable with specific domains
cors_origins = settings.cors_origins_list if settings.cors_origins_list else ["*"]
logger.info(f"CORS origins configured: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,  # Set to False when using wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Heikin Ashi Backtest Server v2.0",
        "status": "running",
        "database": "MongoDB",
        "features": [
            "Daily and Weekly backtesting",
            "Data synchronization",
            "Heikin Ashi calculations",
            "Smart portfolio management"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected"
    }


# Include routers
app.include_router(data.router)
app.include_router(symbols.router)
app.include_router(backtest.router)
app.include_router(advanced_backtest.router)
