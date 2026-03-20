import os
import logging
from flask_login import LoginManager
from pymongo import MongoClient
from services.hybrid_matcher import HybridMatcher
from services.mongo_service import MongoDBManager
from services.constraint_matcher import ConstraintMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

login_manager = LoginManager()

# MongoDB collections (set during init)
db = None
users_collection = None
resumes_collection = None
recruiters_collection = None
jobs_collection = None
applications_collection = None
ai_results_collection = None

# AI/ML components (set during init)
cm = None
manager = None
matcher = None


def init_extensions(app):
    global db, users_collection, resumes_collection, recruiters_collection
    global jobs_collection, applications_collection, ai_results_collection
    global cm, manager, matcher

    # Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    # MongoDB
    client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/'))
    db = client['resume_parser']
    users_collection = db['users']
    resumes_collection = db['parsed_resumes']
    recruiters_collection = db['recruiters']
    jobs_collection = db['jobs']
    applications_collection = db['application_collections']
    ai_results_collection = db['ai_results_collection']

    # Ensure upload folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PARSED_RESUMES_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PARSED_JOB_PATH'], exist_ok=True)

    # AI/ML — no local model needed, uses HF Inference API
    cm = ConstraintMatcher()
    manager = MongoDBManager(os.getenv('MONGO_URI', 'mongodb://localhost:27017/'), "resume_parser")
    matcher = HybridMatcher(constraint_matcher=cm, mongodb_manager=manager)
