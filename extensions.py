import os
import logging
import threading
import uuid
from collections import OrderedDict
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_mail import Mail
from pymongo import MongoClient, ASCENDING
from services.hybrid_matcher import HybridMatcher
from services.mongo_service import MongoDBManager
from services.constraint_matcher import ConstraintMatcher
from services.embedding_service import init_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

login_manager = LoginManager()
csrf = CSRFProtect()
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per hour"])
mail = Mail()

# MongoDB collections (set during init)
db = None
users_collection = None
resumes_collection = None
recruiters_collection = None
jobs_collection = None
applications_collection = None
ai_results_collection = None
embedding_cache_collection = None
candidate_insights_collection = None
insight_chats_collection = None
talent_pool_collection = None
background_tasks_collection = None

# AI/ML components (set during init)
cm = None
manager = None
matcher = None

# Candidate insight services (set during init)
insight_service = None
chat_service = None

# Background task runner
task_runner = None


class TaskRunner:
    """Simple thread-based background task runner for AI engine jobs."""

    def __init__(self, max_results=200):
        self._results = OrderedDict()
        self._lock = threading.Lock()
        self._max_results = max_results

    def submit(self, fn, *args, **kwargs):
        task_id = str(uuid.uuid4())
        with self._lock:
            self._results[task_id] = {'status': 'running', 'result': None, 'error': None}
            # Evict oldest entries if over limit
            while len(self._results) > self._max_results:
                self._results.popitem(last=False)

        def _run():
            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    self._results[task_id] = {'status': 'completed', 'result': result, 'error': None}
            except Exception as e:
                logger.error(f"Background task {task_id} failed: {e}")
                with self._lock:
                    self._results[task_id] = {'status': 'failed', 'result': None, 'error': str(e)}

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return task_id

    def get_status(self, task_id):
        with self._lock:
            return self._results.get(task_id)


def init_extensions(app):
    global db, users_collection, resumes_collection, recruiters_collection
    global jobs_collection, applications_collection, ai_results_collection
    global embedding_cache_collection, candidate_insights_collection, insight_chats_collection
    global talent_pool_collection, background_tasks_collection
    global cm, manager, matcher, task_runner, insight_service, chat_service

    # Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    # CSRF protection
    csrf.init_app(app)

    # Rate limiter
    limiter.init_app(app)

    # Flask-Mail
    mail.init_app(app)
    if not app.config.get('MAIL_USERNAME') or not app.config.get('MAIL_PASSWORD'):
        logger.warning("MAIL_USERNAME/MAIL_PASSWORD not set — email notifications disabled")

    if not app.config.get('GOOGLE_CLIENT_ID') or not app.config.get('GOOGLE_CLIENT_SECRET'):
        logger.warning("GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET not set — Google Calendar integration disabled")

    # MongoDB (single shared client)
    mongo_uri = os.environ.get('MONGO_URI')
    if not mongo_uri:
        raise RuntimeError("MONGO_URI environment variable is required")

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    db = client['resume_parser']
    users_collection = db['users']
    resumes_collection = db['parsed_resumes']
    recruiters_collection = db['recruiters']
    jobs_collection = db['jobs']
    applications_collection = db['application_collections']
    ai_results_collection = db['ai_results_collection']
    embedding_cache_collection = db['embedding_cache']
    candidate_insights_collection = db['candidate_insights']
    insight_chats_collection = db['insight_chats']
    talent_pool_collection = db['talent_pool']
    background_tasks_collection = db['background_tasks']

    # Create indexes for query performance
    _create_indexes()

    # Initialize embedding cache
    init_cache(embedding_cache_collection)

    # Ensure upload folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PARSED_RESUMES_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PARSED_JOB_PATH'], exist_ok=True)

    # AI/ML — share the existing db connection (no duplicate MongoClient)
    cm = ConstraintMatcher()
    manager = MongoDBManager(db)
    matcher = HybridMatcher(constraint_matcher=cm, mongodb_manager=manager)

    # Candidate insight services
    from services.candidate_insight_service import CandidateInsightService
    from services.chat_service import CandidateInsightChat
    insight_service = CandidateInsightService(candidate_insights_collection)
    chat_service = CandidateInsightChat(insight_chats_collection)

    # Background task runner for async AI engine
    task_runner = TaskRunner()


def _create_indexes():
    """Create MongoDB indexes for query performance."""
    try:
        users_collection.create_index([('email', ASCENDING)], unique=True)
        recruiters_collection.create_index([('email', ASCENDING)], unique=True)
        resumes_collection.create_index([('user_id', ASCENDING)])
        jobs_collection.create_index([('recruiter_id', ASCENDING)])
        applications_collection.create_index([('user_id', ASCENDING)])
        applications_collection.create_index([('job_id', ASCENDING)])
        applications_collection.create_index([('resume_id', ASCENDING)])
        applications_collection.create_index(
            [('user_id', ASCENDING), ('job_id', ASCENDING)],
            unique=True
        )
        ai_results_collection.create_index([('job_id', ASCENDING)])
        embedding_cache_collection.create_index([('_k', ASCENDING)], unique=True)
        candidate_insights_collection.create_index(
            [('job_id', ASCENDING), ('resume_id', ASCENDING)], unique=True
        )
        insight_chats_collection.create_index(
            [('job_id', ASCENDING), ('resume_id', ASCENDING), ('recruiter_id', ASCENDING)],
            unique=True
        )
        talent_pool_collection.create_index(
            [('recruiter_id', ASCENDING), ('resume_id', ASCENDING)], unique=True
        )
        talent_pool_collection.create_index([('recruiter_id', ASCENDING), ('tags', ASCENDING)])
        talent_pool_collection.create_index([('recruiter_id', ASCENDING), ('star_rating', ASCENDING)])
        talent_pool_collection.create_index([('recruiter_id', ASCENDING), ('candidate_skills', ASCENDING)])
        background_tasks_collection.create_index([('task_id', ASCENDING)], unique=True)
        background_tasks_collection.create_index(
            [('created_at', ASCENDING)],
            expireAfterSeconds=86400,  # 24h TTL auto-cleanup
        )
        logger.info("MongoDB indexes created successfully")
    except Exception as e:
        logger.warning(f"Index creation warning (may already exist): {e}")
