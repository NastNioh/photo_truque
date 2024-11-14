# database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# URL de la base de données PostgreSQL (modifiez avec vos informations)
DATABASE_URL = "postgresql://postgres:admin@localhost:5432/photo_truque"

# Initialisation
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modèle ImageAnalysis
class ImageAnalysis(Base):
    __tablename__ = "image_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    probability = Column(Float)
    is_truque = Column(Boolean)
    message = Column(String)
    decision = Column(String)
    return_decision = Column(String)

# Crée les tables dans la base de données
Base.metadata.create_all(bind=engine)
