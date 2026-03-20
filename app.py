import os
from flask import Flask
from dotenv import load_dotenv

load_dotenv()


def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
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

    return app


if __name__ == '__main__':
    create_app().run(debug=True, host='0.0.0.0', port=5000)
