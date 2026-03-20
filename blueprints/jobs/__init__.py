from flask import Blueprint

jobs_bp = Blueprint('jobs', __name__)

from blueprints.jobs import routes  # noqa: E402, F401
