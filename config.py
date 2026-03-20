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

    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY')
    if not SECRET_KEY:
        raise RuntimeError(
            "FLASK_SECRET_KEY environment variable is required. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )

    RATELIMIT_STORAGE_URI = "memory://"
    RATELIMIT_DEFAULT = "200/hour"

    # Mail (Gmail SMTP)
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_USERNAME')
