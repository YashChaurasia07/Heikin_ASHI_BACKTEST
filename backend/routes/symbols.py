from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
from models import Symbol
from database import get_database
from datetime import datetime
import pandas as pd
import io
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/symbols", tags=["symbols"])


@router.get("/", response_model=List[Symbol])
async def get_all_symbols():
    """Get all symbols (optimized with limit)"""
    try:
        db = get_database()
        # Limit to prevent huge loads
        symbols = await db.symbols.find().limit(1000).to_list(length=1000)
        return symbols
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add")
async def add_symbol(symbol: str, exchange: str = "NSE"):
    """Add a new symbol"""
    try:
        db = get_database()
        
        # Check if already exists
        existing = await db.symbols.find_one({"symbol": symbol})
        if existing:
            return {"message": f"Symbol {symbol} already exists", "symbol": symbol}
        
        # Add symbol
        symbol_doc = {
            "symbol": symbol,
            "exchange": exchange,
            "active": True,
            "added_date": datetime.utcnow()
        }
        
        await db.symbols.insert_one(symbol_doc)
        
        logger.info(f"Added symbol: {symbol}")
        return {"message": f"Symbol {symbol} added successfully", "symbol": symbol}
        
    except Exception as e:
        logger.error(f"Error adding symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk-add")
async def bulk_add_symbols(symbols: List[str], exchange: str = "NSE"):
    """Add multiple symbols at once"""
    try:
        db = get_database()
        
        added = []
        skipped = []
        
        for symbol in symbols:
            existing = await db.symbols.find_one({"symbol": symbol})
            if existing:
                skipped.append(symbol)
                continue
            
            symbol_doc = {
                "symbol": symbol,
                "exchange": exchange,
                "active": True,
                "added_date": datetime.utcnow()
            }
            
            await db.symbols.insert_one(symbol_doc)
            added.append(symbol)
        
        logger.info(f"Bulk add: {len(added)} added, {len(skipped)} skipped")
        
        return {
            "added": added,
            "skipped": skipped,
            "total_added": len(added),
            "total_skipped": len(skipped)
        }
        
    except Exception as e:
        logger.error(f"Error in bulk add: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-excel")
async def upload_symbols_excel(file: UploadFile = File(...)):
    """Upload symbols from Excel file"""
    try:
        # Read Excel file
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        logger.info(f"Excel columns: {df.columns.tolist()}")
        
        # Try to extract symbols from different possible column names
        symbols = []
        
        if 'Row Labels' in df.columns:
            symbols = df['Row Labels'].dropna().tolist()
        elif 'symbol' in df.columns:
            symbols = df['symbol'].dropna().tolist()
        elif 'Symbol' in df.columns:
            symbols = df['Symbol'].dropna().tolist()
        else:
            # Use first column
            first_col = df.columns[0]
            symbols = df[first_col].dropna().tolist()
        
        # Clean symbols
        symbols = [str(s).strip() for s in symbols if str(s).strip() and not str(s).lower() in ['nan', 'none', '']]
        
        if not symbols:
            raise HTTPException(status_code=400, detail="No symbols found in Excel file")
        
        # Bulk add
        db = get_database()
        added = []
        skipped = []
        
        for symbol in symbols:
            existing = await db.symbols.find_one({"symbol": symbol})
            if existing:
                skipped.append(symbol)
                continue
            
            symbol_doc = {
                "symbol": symbol,
                "exchange": "NSE",
                "active": True,
                "added_date": datetime.utcnow()
            }
            
            await db.symbols.insert_one(symbol_doc)
            added.append(symbol)
        
        logger.info(f"Excel upload: {len(added)} added, {len(skipped)} skipped")
        
        return {
            "added": added,
            "skipped": skipped,
            "total_added": len(added),
            "total_skipped": len(skipped)
        }
        
    except Exception as e:
        logger.error(f"Error uploading Excel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{symbol}")
async def delete_symbol(symbol: str):
    """Delete a symbol (soft delete - mark as inactive)"""
    try:
        db = get_database()
        
        result = await db.symbols.update_one(
            {"symbol": symbol},
            {"$set": {"active": False}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        logger.info(f"Deactivated symbol: {symbol}")
        return {"message": f"Symbol {symbol} deactivated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))
