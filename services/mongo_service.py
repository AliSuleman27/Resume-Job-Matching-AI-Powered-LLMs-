from pymongo import MongoClient
from bson import ObjectId
from typing import List, Optional, Dict, Tuple
import re
from enum import Enum
from dataclasses import dataclass
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBManager:
    def __init__(self, connection_string, db_name):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.jobs_collection = self.db['jobs']
        self.resumes_collection = self.db['parsed_resumes']
        self.applications_collection = self.db['application_collections']  # Added applications collection
    
    def get_all_jobs(self):
        return list(self.jobs_collection.find({}))
    
    def get_all_resumes(self):
        return list(self.resumes_collection.find({}))
    
    def get_job_by_id(self, job_id):
        try:
            return self.jobs_collection.find_one({"_id": ObjectId(job_id)})
        except:
            return None
    
    def get_resume_by_id(self, resume_id):
        try:
            return self.resumes_collection.find_one({"_id": ObjectId(resume_id)})
        except:
            return None
    
    def get_job_applicants(self, job_id):
        """Get all applicants for a specific job"""
        try:
            applications = list(self.applications_collection.find({"job_id": ObjectId(job_id)}))
            return applications
        except Exception as e:
            logger.error(f"Error fetching applicants for job {job_id}: {e}")
            return []
    
    def get_user_by_id(self, user_id):
        """Get user information by user_id"""
        try:
            return self.db['users'].find_one({"_id": ObjectId(user_id)})
        except:
            return None
    
    def get_job_by_recruiter_id(self, recruiter_id):
        try:
            return list(self.jobs_collection.find({"recruiter_id": ObjectId(recruiter_id)}))
        except:
            return []
    
    def get_resume_by_user_id(self, user_id):
        try:
            return list(self.resumes_collection.find({"user_id": ObjectId(user_id)}))
        except:
            return []
    
    def close(self):
        self.client.close()
