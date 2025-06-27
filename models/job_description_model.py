from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field
from uuid import UUID
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
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    zip_code: Optional[str]
    remote: Optional[bool]


class Salary(BaseModel):
    currency: Optional[str]
    min: Optional[float]
    max: Optional[float]
    period: Optional[SalaryPeriod]
    is_estimated: Optional[bool]


class EducationRequirement(BaseModel):
    degree: Optional[str]
    field_of_study: Optional[str]
    level: Optional[EducationLevel]


class ExperienceYears(BaseModel):
    min: Optional[float]
    max: Optional[float]


class Qualifications(BaseModel):
    education: Optional[List[EducationRequirement]] = []
    experience_years: Optional[ExperienceYears]
    certifications: Optional[List[str]] = []


class Skills(BaseModel):
    mandatory: Optional[List[str]] = []
    optional: Optional[List[str]] = []
    tools: Optional[List[str]] = []


class LanguageRequirement(BaseModel):
    language: str
    proficiency: Optional[LanguageProficiency]


class CompanyInfo(BaseModel):
    name: Optional[str]
    website: Optional[str]
    description: Optional[str]


class Analytics(BaseModel):
    views: Optional[int]
    applications: Optional[int]


class Metadata(BaseModel):
    created_at: Optional[str]
    updated_at: Optional[str]
    created_by_user_id: Optional[str]
    source: Optional[str]


class JobDescription(BaseModel):
    job_id: UUID
    title: str
    description: Optional[str]
    summary: Optional[str]
    employment_type: Optional[EmploymentType]
    industry: Optional[str]
    department: Optional[str]
    function: Optional[str]
    job_level: Optional[JobLevel]
    locations: Optional[List[Location]] = []
    is_remote: Optional[bool]
    is_hybrid: Optional[bool]
    is_onsite: Optional[bool]
    application_url: Optional[str]  # Can change to HttpUrl if strictly URL
    posting_date: Optional[str]
    closing_date: Optional[str]
    salary: Optional[Salary]
    benefits: Optional[List[str]] = []
    qualifications: Optional[Qualifications]
    skills: Optional[Skills]
    languages: Optional[List[LanguageRequirement]] = []
    responsibilities: Optional[List[str]] = []
    requirements: Optional[List[str]] = []
    nice_to_have: Optional[List[str]] = []
    company: Optional[CompanyInfo]
    analytics: Optional[Analytics]
    metadata: Optional[Metadata]
