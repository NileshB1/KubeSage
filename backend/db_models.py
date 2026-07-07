"""

SQLAlchemy ORM schemas for tracking historical Kubernetes incidents, generated diagnostic 
reports, embedding mappings, and experiment runs
Relation in persistence in Postgres.
"""

from datetime import datetime
from uuid import uuid4


from sqlalchemy import (
    Column, String, Text, Float, Integer, 
    DateTime,
    JSON, ForeignKey, 
    Boolean, create_engine, Index,
)

from sqlalchemy.dialects.postgresql import UUID, ARRAY



from sqlalchemy.orm import declarative_base, relationship, sessionmaker


Base = declarative_base()


class Incident(Base):
    """
    Stores raw Kubernetes incident data.
    Each incident is uniquely identified and linked to reports.
    """
    __tablename__ = "incidents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    incident_id=Column(String(50), unique=True, nullable=False, index=True)  # e.g., INC-1045
    title = Column(String(500), nullable=False)
    description=Column(Text, nullable=False)
    severity = Column(String(20), nullable=False, index=True)  # Critical, High, Medium, Low
    root_cause = Column(String(500), nullable=True)
    resolution = Column(Text, nullable=True)
    evidence = Column(JSON, nullable=True)  # List of evidence items
    affected_services = Column(ARRAY(String), nullable=True)
    affected_pods = Column(ARRAY(String), nullable=True)
    incident_type=Column(String(100), nullable=False, index=True)
    #For example OOMKilled, CrashLoopBackOff, ImagePullBackOff, etc.
    source = Column(String(50), nullable=False)  # "synthetic" or "real"
    timestamp=Column(DateTime, default=datetime.utcnow, index=True)
    metadata_=Column("metadata", JSON, nullable=True)

    # Relationships
    reports=relationship("Report", back_populates="incident", cascade="all, delete-orphan")
    embedding=relationship("EmbeddingMetadata", back_populates="incident", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_incident_type_severity", "incident_type", "severity"),
        Index("idx_incident_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<Incident {self.incident_id} [{self.severity}] {self.incident_type}>"


class EmbeddingMetadata(Base):
    """
    Stores embedding vector metadata
    """
    __tablename__ = "embedding_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    incident_id=Column(UUID(as_uuid=True), ForeignKey("incidents.id"), unique=True, nullable=False)
    chroma_id = Column(String(100), unique=True, nullable=False, index=True)
    model_name=Column(String(100), nullable=False)  # e.g., all-MiniLM-L6-v2
    dimension=Column(Integer, nullable=False)

    text_chunk = Column(Text, nullable=True)  # The text that was embedded
    created_at = Column(DateTime, default=datetime.utcnow)

    incident = relationship("Incident", back_populates="embedding")

    def __repr__(self) -> str:
        return f"<Embedding {self.model_name} → {self.incident_id}>"


class Report(Base):
    """
    Stores generated incident investigation reports.
    """
    __tablename__ = "reports"

    id=Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    incident_id=Column(UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False, index=True)
    report_text = Column(Text, nullable=False)
    root_cause = Column(String(500), nullable=True)
    confidence_score = Column(Float, nullable=True)
    rag_enabled = Column(Boolean, default=True)
    llm_model = Column(String(100), nullable=False)
    prompt_template = Column(String(50), nullable=True)
    retrieved_incident_ids = Column(ARRAY(String), nullable=True)  # IDs of retrieved incidents
    generated_at=Column(DateTime, default=datetime.utcnow)
    metadata_=Column("metadata", JSON, nullable=True)

    incident = relationship("Incident", back_populates="reports")

    __table_args__ = (
        Index("idx_report_incident_id", "incident_id"),
    )

    def __repr__(self) -> str:
        return f"<Report for {self.incident_id} [{self.llm_model}]>"


class ExperimentResult(Base):
    """
    Stores results from evaluation experiments.
    """
    __tablename__="experiment_results"

    id=Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    experiment_name=Column(String(200), nullable=False, index=True)
    experiment_config=Column(JSON, nullable=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=True)
    top_k = Column(Integer, nullable=True)
    rag_enabled=Column(Boolean, nullable=True)
    run_at=Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Experiment {self.experiment_name}: {self.metric_name}={self.metric_value}>"


def get_engine(database_url: str | None = None):
    """Create SQLAlchemy engine"""
    
    from backend.config import settings
    url = database_url or settings.postgres_url
    return create_engine(url, echo=False, pool_size=5, max_overflow=10)


def get_session(engine=None):
    """create a new database session"""
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(engine=None):
    """Create all tables if they don't 
    exist"""
    if engine is None:
        engine = get_engine()
        
    Base.metadata.create_all(engine)
