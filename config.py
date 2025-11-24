import os
from urllib.parse import quote_plus

class Config:
    """Database configuration"""
    
    # Azure PostgreSQL bağlantı bilgileri
    POSTGRES_USER = 'nuvotekadmin'
    POSTGRES_PASSWORD = 'nwe@2025'  
    POSTGRES_HOST = 'nuvotekserver.postgres.database.azure.com'
    POSTGRES_PORT = '5432'
    POSTGRES_DB = 'pilz_reports'
    
    # Şifreyi URL-safe hale getir
    ENCODED_PASSWORD = quote_plus(POSTGRES_PASSWORD)
    
    # SQLAlchemy connection string (Azure için SSL gerekli)
    SQLALCHEMY_DATABASE_URI = f'postgresql://{POSTGRES_USER}:{ENCODED_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}?sslmode=require'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False