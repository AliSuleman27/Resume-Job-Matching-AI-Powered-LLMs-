from typing import List, Optional
from pydantic import BaseModel


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
    field: Optional[str]
    institution: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    grade: Optional[str]
    location: Optional[str]
    courses: List[str]


class Experience(BaseModel):
    job_title: str
    company: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    employment_type: Optional[str]
    location: Optional[str]
    responsibilities: List[str]
    skills_used: List[str]


class Project(BaseModel):
    title: str
    description: Optional[str]
    technologies: List[str]
    role: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    link: Optional[str]


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
    summary: Optional[str]
    skills: List[Skill]
    education: List[Education]
    experience: List[Experience]
    projects: List[Project]
    certifications: Optional[List[Certification]] = []
    publications: Optional[List[Publication]] = []
    languages: List[Language]
    interests: Optional[List[str]] = []
    volunteer_experience: Optional[List[VolunteerExperience]] = []
    awards: Optional[List[Award]] = []
    links: Optional[List[Link]] = []
