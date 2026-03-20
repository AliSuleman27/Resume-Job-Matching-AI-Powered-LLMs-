from flask import Blueprint

ai_engine_bp = Blueprint('ai_engine', __name__, url_prefix='/recruiter')

from blueprints.ai_engine import routes  # noqa: E402, F401
