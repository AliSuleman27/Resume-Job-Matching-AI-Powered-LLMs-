from typing import List, Optional
from pydantic import BaseModel, field_validator


class Address(BaseModel):
    street: Optional[str]
    city: Optional[str]
    state: Optional[str]
    zip_code: Optional[str]
    country: Optional[str]


class BasicInfo(BaseModel):
    full_name: str
    current_title: Optional[str]
    gender: Optional[str]
    date_of_birth: Optional[str]
    nationality: Optional[str]


class ContactInfo(BaseModel):
    email: Optional[str]
    phone: Optional[str]
    address: Optional[Address]
    linkedin: Optional[str]
    github: Optional[str]
    portfolio: Optional[str]


class Skill(BaseModel):
    skill_name: str
    proficiency: Optional[str]
    category: Optional[str]
    years_of_experience: Optional[int]


class Education(BaseModel):
    degree: str
    field: Optional[str] = None
    institution: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    grade: Optional[str] = None
    location: Optional[str] = None
    courses: Optional[List[str]] = []

    @field_validator('courses', mode='before')
    @classmethod
    def coerce_courses(cls, v):
        return v if v is not None else []


class Experience(BaseModel):
    job_title: str
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    employment_type: Optional[str] = None
    location: Optional[str] = None
    responsibilities: Optional[List[str]] = []
    skills_used: Optional[List[str]] = []

    @field_validator('responsibilities', 'skills_used', mode='before')
    @classmethod
    def coerce_lists(cls, v):
        return v if v is not None else []


class Project(BaseModel):
    title: str
    description: Optional[str] = None
    technologies: Optional[List[str]] = []
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    link: Optional[str] = None

    @field_validator('technologies', mode='before')
    @classmethod
    def coerce_technologies(cls, v):
        return v if v is not None else []


class Certification(BaseModel):
    title: str
    issuer: Optional[str]
    issue_date: Optional[str]
    expiration_date: Optional[str]
    credential_id: Optional[str]
    url: Optional[str]


class Publication(BaseModel):
    title: str
    journal: Optional[str]
    authors: Optional[List[str]]
    year: Optional[int]
    doi: Optional[str]
    url: Optional[str]


class Language(BaseModel):
    language: str
    proficiency: Optional[str]


class VolunteerExperience(BaseModel):
    role: str
    organization: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    description: Optional[str]


class Award(BaseModel):
    title: str
    issuer: Optional[str]
    date: Optional[str]
    description: Optional[str]


class Link(BaseModel):
    label: Optional[str]
    url: Optional[str]


class Resume(BaseModel):
    basic_info: BasicInfo
    contact_info: ContactInfo
    summary: Optional[str] = None
    skills: Optional[List[Skill]] = []
    education: Optional[List[Education]] = []
    experience: Optional[List[Experience]] = []
    projects: Optional[List[Project]] = []
    certifications: Optional[List[Certification]] = []
    publications: Optional[List[Publication]] = []
    languages: Optional[List[Language]] = []
    interests: Optional[List[str]] = []
    volunteer_experience: Optional[List[VolunteerExperience]] = []
    awards: Optional[List[Award]] = []
    links: Optional[List[Link]] = []

    @field_validator('skills', 'education', 'experience', 'projects',
                     'certifications', 'publications', 'languages',
                     'interests', 'volunteer_experience', 'awards', 'links',
                     mode='before')
    @classmethod
    def coerce_none_to_list(cls, v):
        return v if v is not None else []
