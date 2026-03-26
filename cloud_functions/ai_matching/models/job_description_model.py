from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field
from uuid import UUID, uuid4
from enum import Enum


class EmploymentType(str, Enum):
    full_time = "full_time"
    part_time = "part_time"
    contract = "contract"
    temporary = "temporary"
    internship = "internship"
    freelance = "freelance"


class JobLevel(str, Enum):
    entry = "entry"
    mid = "mid"
    senior = "senior"
    lead = "lead"
    manager = "manager"
    director = "director"
    executive = "executive"


class SalaryPeriod(str, Enum):
    hour = "hour"
    day = "day"
    week = "week"
    month = "month"
    year = "year"


class EducationLevel(str, Enum):
    high_school = "high_school"
    associate = "associate"
    bachelor = "bachelor"
    master = "master"
    phd = "phd"
    other = "other"


class LanguageProficiency(str, Enum):
    basic = "basic"
    conversational = "conversational"
    fluent = "fluent"
    native = "native"


class Location(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    zip_code: Optional[str] = None
    remote: Optional[bool] = None


class Salary(BaseModel):
    currency: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    period: Optional[SalaryPeriod] = None
    is_estimated: Optional[bool] = None


class EducationRequirement(BaseModel):
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    level: Optional[EducationLevel] = None


class ExperienceYears(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None


class Qualifications(BaseModel):
    education: Optional[List[EducationRequirement]] = []
    experience_years: Optional[ExperienceYears] = None
    certifications: Optional[List[str]] = []


class Skills(BaseModel):
    mandatory: Optional[List[str]] = []
    optional: Optional[List[str]] = []
    tools: Optional[List[str]] = []


class LanguageRequirement(BaseModel):
    language: str
    proficiency: Optional[LanguageProficiency] = None


class CompanyInfo(BaseModel):
    name: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None


class Analytics(BaseModel):
    views: Optional[int] = None
    applications: Optional[int] = None


class Metadata(BaseModel):
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    created_by_user_id: Optional[str] = None
    source: Optional[str] = None


class JobDescription(BaseModel):
    job_id: UUID = Field(default_factory=uuid4)
    title: str
    description: Optional[str] = None
    summary: Optional[str] = None
    employment_type: Optional[EmploymentType] = None
    industry: Optional[str] = None
    department: Optional[str] = None
    function: Optional[str] = None
    job_level: Optional[JobLevel] = None
    locations: Optional[List[Location]] = []
    is_remote: Optional[bool] = None
    is_hybrid: Optional[bool] = None
    is_onsite: Optional[bool] = None
    application_url: Optional[str] = None
    posting_date: Optional[str] = None
    closing_date: Optional[str] = None
    salary: Optional[Salary] = None
    benefits: Optional[List[str]] = []
    qualifications: Optional[Qualifications] = None
    skills: Optional[Skills] = None
    languages: Optional[List[LanguageRequirement]] = []
    responsibilities: Optional[List[str]] = []
    requirements: Optional[List[str]] = []
    nice_to_have: Optional[List[str]] = []
    company: Optional[CompanyInfo] = None
    analytics: Optional[Analytics] = None
    metadata: Optional[Metadata] = None
