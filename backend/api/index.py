from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
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

# Create FastAPI app
app = FastAPI(
    title="Heikin Ashi Backtest Server",
    description="Professional backtesting API with MongoDB storage",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection flag
_db_connected = False


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global _db_connected
    if not _db_connected:
        logger.info("Starting Heikin Ashi Backtest Server...")
        await connect_to_mongo()
        _db_connected = True
        logger.info("Server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down server...")
    await close_mongo_connection()
    logger.info("Server stopped")


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

# Mangum handler for Vercel
handler = Mangum(app)
