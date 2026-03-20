import os
from datetime import datetime, timezone
from flask import Flask
from dotenv import load_dotenv

load_dotenv()


def create_app():
    app = Flask(__name__)

    secret = os.environ.get('FLASK_SECRET_KEY')
    if not secret:
        raise RuntimeError(
            "FLASK_SECRET_KEY environment variable is required. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    app.secret_key = secret
    app.config.from_object('config.Config')

    from extensions import init_extensions
    init_extensions(app)

    # Import user_models to register the user_loader
    import blueprints.auth.user_models  # noqa: F401

    from blueprints.auth import auth_bp
    from blueprints.applicant import applicant_bp
    from blueprints.recruiter import recruiter_bp
    from blueprints.jobs import jobs_bp
    from blueprints.ai_engine import ai_engine_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(applicant_bp)
    app.register_blueprint(recruiter_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(ai_engine_bp)

    @app.context_processor
    def inject_now():
        return {'now': datetime.now(timezone.utc)}

    @app.after_request
    def set_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
        if not app.debug:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response

    return app


if __name__ == '__main__':
    create_app().run(debug=False, host='127.0.0.1', port=5000)
