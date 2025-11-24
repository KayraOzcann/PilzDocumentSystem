from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from config import Config

# SQLAlchemy instance
db = SQLAlchemy()

def init_db(app):
    """Initialize database with Flask app"""
    app.config['SQLALCHEMY_DATABASE_URI'] = Config.SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = Config.SQLALCHEMY_TRACK_MODIFICATIONS
    app.config['SQLALCHEMY_ECHO'] = Config.SQLALCHEMY_ECHO
    
    db.init_app(app)
    
    return db

def create_tables(app):
    """Create all tables"""
    with app.app_context():
        db.create_all()
        print("✅ Tüm tablolar oluşturuldu!")

def drop_tables(app):
    """Drop all tables (DEV ONLY!)"""
    with app.app_context():
        db.drop_all()
        print("❌ Tüm tablolar silindi!")