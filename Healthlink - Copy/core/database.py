"""
Database module for HealthLink.
Uses SQLAlchemy with SQLite for local storage.
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Date, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from config.settings import Settings


logger = logging.getLogger("healthlink.database")

Base = declarative_base()


class DoctorModel(Base):
    """Doctor database model."""
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    specialty = Column(String(100), nullable=False)
    experience_years = Column(Integer, nullable=False)
    rating = Column(Float, nullable=False)
    availability = Column(String(100), nullable=False)
    location = Column(String(200), nullable=False)
    email = Column(String(200), nullable=True)
    phone = Column(String(20), nullable=True)
    qualifications = Column(String(500), nullable=True)
    languages = Column(String(200), nullable=True)
    consultation_type = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AppointmentModel(Base):
    """Appointment database model."""
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False)
    doctor_id = Column(Integer, nullable=False)
    appointment_date = Column(Date, nullable=False)
    appointment_time = Column(String(10), nullable=False)
    status = Column(String(20), nullable=False, default="scheduled")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SessionLogModel(Base):
    """Session log for tracking user interactions."""
    __tablename__ = "session_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False)
    request_id = Column(String(100), nullable=False)
    user_input = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Database manager for handling connections and sessions."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = create_engine(
            settings.database_url,
            echo=settings.db_echo,
            connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._initialized = False

    def initialize_database(self) -> None:
        """Create all tables if they don't exist."""
        if not self._initialized:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified")
            self._initialized = True

    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope for database operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}", exc_info=True)
            raise
        finally:
            session.close()


_db_manager: Optional[DatabaseManager] = None


def get_db_manager(settings: Settings) -> DatabaseManager:
    """
    FastAPI dependency for database manager.

    Args:
        settings: Application settings

    Returns:
        Database manager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(settings)
        _db_manager.initialize_database()
    return _db_manager


def get_db_session(settings: Settings) -> Session:
    """
    FastAPI dependency for database session.

    Args:
        settings: Application settings

    Yields:
        Database session
    """
    db_manager = get_db_manager(settings)
    session = db_manager.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_all_doctors(session: Session) -> List[DoctorModel]:
    """
    Get all doctors from database.

    Args:
        session: Database session

    Returns:
        List of doctor models
    """
    return session.query(DoctorModel).all()


def get_doctors_by_specialty(session: Session, specialty: str) -> List[DoctorModel]:
    """
    Get doctors by specialty.

    Args:
        session: Database session
        specialty: Medical specialty

    Returns:
        List of matching doctors
    """
    return session.query(DoctorModel).filter(
        DoctorModel.specialty.ilike(f"%{specialty}%")
    ).all()


def get_doctor_by_id(session: Session, doctor_id: int) -> Optional[DoctorModel]:
    """
    Get doctor by ID.

    Args:
        session: Database session
        doctor_id: Doctor ID

    Returns:
        Doctor model or None
    """
    return session.query(DoctorModel).filter(DoctorModel.id == doctor_id).first()


def create_appointment(
    session: Session,
    user_id: str,
    doctor_id: int,
    appointment_date: date,
    appointment_time: str,
    notes: Optional[str] = None
) -> AppointmentModel:
    """
    Create a new appointment.

    Args:
        session: Database session
        user_id: User identifier
        doctor_id: Doctor ID
        appointment_date: Appointment date
        appointment_time: Appointment time
        notes: Optional notes

    Returns:
        Created appointment model
    """
    appointment = AppointmentModel(
        user_id=user_id,
        doctor_id=doctor_id,
        appointment_date=appointment_date,
        appointment_time=appointment_time,
        status="scheduled",
        notes=notes
    )
    session.add(appointment)
    session.commit()
    session.refresh(appointment)

    logger.info(f"Created appointment {appointment.id} for user {user_id}")
    return appointment


def get_appointments_by_user(session: Session, user_id: str) -> List[AppointmentModel]:
    """
    Get all appointments for a user.

    Args:
        session: Database session
        user_id: User identifier

    Returns:
        List of appointments
    """
    return session.query(AppointmentModel).filter(
        AppointmentModel.user_id == user_id
    ).order_by(AppointmentModel.appointment_date.desc()).all()


def log_session(
    session: Session,
    user_id: str,
    request_id: str,
    user_input: str,
    response: Optional[str] = None
) -> SessionLogModel:
    """
    Log a user session interaction.

    Args:
        session: Database session
        user_id: User identifier
        request_id: Request identifier
        user_input: User's input
        response: System response

    Returns:
        Created session log
    """
    log = SessionLogModel(
        user_id=user_id,
        request_id=request_id,
        user_input=user_input,
        response=response
    )
    session.add(log)
    session.commit()
    session.refresh(log)

    logger.debug(f"Logged session for user {user_id}, request {request_id}")
    return log


def seed_doctors(session: Session, doctors_data: List[Dict[str, Any]]) -> None:
    """
    Seed database with initial doctor data.

    Args:
        session: Database session
        doctors_data: List of doctor dictionaries
    """
    existing_count = session.query(DoctorModel).count()
    if existing_count > 0:
        logger.info(f"Database already contains {existing_count} doctors, skipping seed")
        return

    for data in doctors_data:
        doctor = DoctorModel(**data)
        session.add(doctor)

    session.commit()
    logger.info(f"Seeded database with {len(doctors_data)} doctors")
