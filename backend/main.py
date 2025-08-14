# backend/main.py
import os
import shutil
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Boolean, Float, create_engine, Text
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

# -----------------
# Config
# -----------------
DB_URL = os.getenv("DVCR_DB", "sqlite:///./dvcr.db")
UPLOAD_DIR = os.getenv("DVCR_UPLOAD_DIR", "uploads")
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------
# DB setup
# -----------------
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# -----------------
# Models
# -----------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    role = Column(String, nullable=False)  # 'driver' | 'manager' | 'mechanic'

class Truck(Base):
    __tablename__ = "trucks"
    id = Column(Integer, primary_key=True)
    number = Column(String, unique=True, index=True, nullable=False)
    vin = Column(String, nullable=True)
    active = Column(Boolean, default=True)

    reports = relationship("Report", back_populates="truck")

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)
    truck_id = Column(Integer, ForeignKey("trucks.id"), nullable=False)
    driver_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    odometer = Column(Integer, nullable=True)
    status = Column(String, default="OPEN")  # OPEN, IN_PROGRESS, RESOLVED, REJECTED
    summary = Column(Text, nullable=True)

    truck = relationship("Truck", back_populates="reports")
    driver = relationship("User")
    defects = relationship("Defect", back_populates="report", cascade="all, delete-orphan")
    notes = relationship("Note", back_populates="report", cascade="all, delete-orphan")

class Defect(Base):
    __tablename__ = "defects"
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    component = Column(String, nullable=False)
    severity = Column(String, default="minor")  # minor | major | OOS
    description = Column(Text, nullable=True)
    x = Column(Float, nullable=True)  # 0..1 relative position on layout
    y = Column(Float, nullable=True)
    resolved = Column(Boolean, default=False)
    resolved_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    report = relationship("Report", back_populates="defects")
    photos = relationship("Photo", back_populates="defect", cascade="all, delete-orphan")
    resolved_by = relationship("User")

class Photo(Base):
    __tablename__ = "photos"
    id = Column(Integer, primary_key=True)
    defect_id = Column(Integer, ForeignKey("defects.id"), nullable=False)
    path = Column(String, unique=True, nullable=False)
    caption = Column(String, nullable=True)

    defect = relationship("Defect", back_populates="photos")

class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    report = relationship("Report", back_populates="notes")
    author = relationship("User")

Base.metadata.create_all(bind=engine)

# -----------------
# Schemas
# -----------------
class UserOut(BaseModel):
    id: int
    name: str
    email: str
    role: str
    class Config:
        orm_mode = True

class TruckIn(BaseModel):
    number: str
    vin: Optional[str] = None
    active: bool = True

class TruckOut(TruckIn):
    id: int
    class Config:
        orm_mode = True

class DefectIn(BaseModel):
    component: str
    severity: str = Field(default="minor")
    description: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None

class DefectOut(DefectIn):
    id: int
    resolved: bool
    resolved_by_id: Optional[int] = None
    resolved_at: Optional[datetime] = None
    class Config:
        orm_mode = True

class PhotoOut(BaseModel):
    id: int
    path: str
    caption: Optional[str]
    class Config:
        orm_mode = True

class NoteIn(BaseModel):
    text: str

class NoteOut(BaseModel):
    id: int
    author: UserOut
    text: str
    created_at: datetime
    class Config:
        orm_mode = True

class ReportIn(BaseModel):
    odometer: Optional[int] = None
    summary: Optional[str] = None

class ReportOut(BaseModel):
    id: int
    truck: TruckOut
    driver: UserOut
    created_at: datetime
    odometer: Optional[int]
    status: str
    summary: Optional[str]
    defects: List[DefectOut] = []
    notes: List[NoteOut] = []
    class Config:
        orm_mode = True

# -----------------
# FastAPI app
# -----------------
app = FastAPI(title="DVCR API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files (dev only)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Dependency: DB session

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency: current user via header (MVP)

def get_current_user(db: Session = Depends(get_db), x_user_id: Optional[str] = None):
    # FastAPI injects headers by parameter name; but we explicitly read from request later if needed.
    # We’ll fallback to reading from the request in endpoints.
    return None

from fastapi import Request

async def require_user(request: Request, db: Session = Depends(get_db)) -> User:
    user_id = request.headers.get("x-user-id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing x-user-id header (MVP auth)")
    user = db.get(User, int(user_id))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Utility role check

def require_role(user: User, roles: List[str]):
    if user.role not in roles:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

# -----------------
# Seed: create demo users if none
# -----------------
with SessionLocal() as db:
    if db.query(User).count() == 0:
        db.add_all([
            User(name="Alice Driver", email="driver@example.com", role="driver"),
            User(name="Manny Manager", email="manager@example.com", role="manager"),
            User(name="Mec McWrench", email="mechanic@example.com", role="mechanic"),
        ])
        if db.query(Truck).count() == 0:
            db.add_all([
                Truck(number="78014", vin="VIN78014"),
                Truck(number="78988", vin="VIN78988"),
            ])
        db.commit()

# -----------------
# Auth (MVP)
# -----------------
class LoginIn(BaseModel):
    email: str

@app.post("/auth/login", response_model=UserOut)
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(401, "Unknown email")
    return user

@app.get("/me", response_model=UserOut)
async def me(user: User = Depends(require_user)):
    return user

# -----------------
# Trucks
# -----------------
@app.get("/trucks", response_model=List[TruckOut])
def list_trucks(db: Session = Depends(get_db)):
    return db.query(Truck).order_by(Truck.number).all()

@app.post("/trucks", response_model=TruckOut)
def create_truck(payload: TruckIn, user: User = Depends(require_user), db: Session = Depends(get_db)):
    require_role(user, ["manager"])  # Only managers can create trucks (MVP)
    t = Truck(number=payload.number, vin=payload.vin, active=payload.active)
    db.add(t)
    db.commit()
    db.refresh(t)
    return t

@app.get("/trucks/{truck_id}", response_model=TruckOut)
def get_truck(truck_id: int, db: Session = Depends(get_db)):
    t = db.get(Truck, truck_id)
    if not t:
        raise HTTPException(404, "Truck not found")
    return t

# -----------------
# Reports
# -----------------
@app.get("/trucks/{truck_id}/reports", response_model=List[ReportOut])
def list_reports(truck_id: int, db: Session = Depends(get_db)):
    return (
        db.query(Report)
        .filter(Report.truck_id == truck_id)
        .order_by(Report.created_at.desc())
        .all()
    )

@app.post("/trucks/{truck_id}/reports", response_model=ReportOut)
def create_report(truck_id: int, payload: ReportIn, user: User = Depends(require_user), db: Session = Depends(get_db)):
    truck = db.get(Truck, truck_id)
    if not truck:
        raise HTTPException(404, "Truck not found")
    # Any role can open a report in MVP; in real app, enforce driver-only
    r = Report(truck_id=truck_id, driver_id=user.id, odometer=payload.odometer, summary=payload.summary)
    db.add(r)
    db.commit()
    db.refresh(r)
    return r

@app.get("/reports/{report_id}", response_model=ReportOut)
def get_report(report_id: int, db: Session = Depends(get_db)):
    r = db.get(Report, report_id)
    if not r:
        raise HTTPException(404, "Report not found")
    return r

class ReportPatch(BaseModel):
    status: Optional[str] = None
    summary: Optional[str] = None
    odometer: Optional[int] = None

@app.patch("/reports/{report_id}", response_model=ReportOut)
def patch_report(report_id: int, payload: ReportPatch, user: User = Depends(require_user), db: Session = Depends(get_db)):
    r = db.get(Report, report_id)
    if not r:
        raise HTTPException(404, "Report not found")
    # Managers/mechanics can change status; drivers can update summary/odometer
    if payload.status is not None:
        require_role(user, ["manager", "mechanic"])
        r.status = payload.status
    if payload.summary is not None:
        r.summary = payload.summary
    if payload.odometer is not None:
        r.odometer = payload.odometer
    db.commit()
    db.refresh(r)
    return r

# -----------------
# Notes
# -----------------
@app.post("/reports/{report_id}/notes", response_model=NoteOut)
def add_note(report_id: int, note: NoteIn, user: User = Depends(require_user), db: Session = Depends(get_db)):
    r = db.get(Report, report_id)
    if not r:
        raise HTTPException(404, "Report not found")
    n = Note(report_id=report_id, author_id=user.id, text=note.text)
    db.add(n)
    db.commit()
    db.refresh(n)
    return n

# -----------------
# Defects & Photos
# -----------------
@app.post("/reports/{report_id}/defects", response_model=DefectOut)
def add_defect(report_id: int, d: DefectIn, user: User = Depends(require_user), db: Session = Depends(get_db)):
    r = db.get(Report, report_id)
    if not r:
        raise HTTPException(404, "Report not found")
    defect = Defect(report_id=report_id, component=d.component, severity=d.severity, description=d.description, x=d.x, y=d.y)
    db.add(defect)
    db.commit()
    db.refresh(defect)
    return defect

class DefectPatch(BaseModel):
    description: Optional[str] = None
    resolved: Optional[bool] = None

@app.patch("/defects/{defect_id}", response_model=DefectOut)
def patch_defect(defect_id: int, payload: DefectPatch, user: User = Depends(require_user), db: Session = Depends(get_db)):
    d = db.get(Defect, defect_id)
    if not d:
        raise HTTPException(404, "Defect not found")
    if payload.description is not None:
        d.description = payload.description
    if payload.resolved is not None:
        # Only mechanic (or manager) can mark resolved
        require_role(user, ["mechanic", "manager"])
        d.resolved = payload.resolved
        d.resolved_by_id = user.id if payload.resolved else None
        d.resolved_at = datetime.utcnow() if payload.resolved else None
    db.commit()
    db.refresh(d)
    return d

@app.post("/defects/{defect_id}/photos", response_model=List[PhotoOut])
async def upload_photos(defect_id: int, files: List[UploadFile] = File(...), captions: Optional[str] = Form(None), user: User = Depends(require_user), db: Session = Depends(get_db)):
    d = db.get(Defect, defect_id)
    if not d:
        raise HTTPException(404, "Defect not found")

    saved: List[Photo] = []
    for f in files:
        # Ensure unique filename
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        safe_name = f"{defect_id}_{ts}_{f.filename}"
        out_path = os.path.join(UPLOAD_DIR, safe_name)
        with open(out_path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        p = Photo(defect_id=defect_id, path=f"/uploads/{safe_name}", caption=captions)
        db.add(p)
        saved.append(p)
    db.commit()
    for p in saved:
        db.refresh(p)
    return saved

# Health
@app.get("/health")
def health():
    return {"ok": True}
