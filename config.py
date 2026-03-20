import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

class Config:
    UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'uploads')
    PARSED_RESUMES_FOLDER = os.path.join(tempfile.gettempdir(), 'parsed_resumes')
    PARSED_JOB_PATH = os.path.join(tempfile.gettempdir(), 'parsed_jobs')
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
