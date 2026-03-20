from flask import Blueprint

applicant_bp = Blueprint('applicant', __name__)

from blueprints.applicant import routes  # noqa: E402, F401
