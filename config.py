import os
from urllib.parse import quote_plus

class Config:
    """Database configuration"""
    
    # ✅ Environment variable'lardan al
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT')
    POSTGRES_DB = os.getenv('POSTGRES_DB')
    
    # Kontrol: Kritik değerler boş mu?
    if not POSTGRES_PASSWORD:
        raise ValueError("❌ POSTGRES_PASSWORD environment variable bulunamadı!")
    
    # Şifreyi URL-safe hale getir
    ENCODED_PASSWORD = quote_plus(POSTGRES_PASSWORD) if POSTGRES_PASSWORD else ''
    
    # SQLAlchemy connection string (Azure için SSL gerekli)
    SQLALCHEMY_DATABASE_URI = f'postgresql://{POSTGRES_USER}:{ENCODED_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}?sslmode=require'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False