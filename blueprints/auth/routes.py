import re
import datetime
from datetime import timezone
from flask import render_template, request, redirect, url_for, flash
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from blueprints.auth import auth_bp
from blueprints.auth.user_models import User, Recruiter
from extensions import users_collection, recruiters_collection, limiter


def validate_password(password):
    """Validate password meets minimum security requirements.
    Returns a list of error messages, empty if valid."""
    errors = []
    if len(password) < 8:
        errors.append('Password must be at least 8 characters long')
    if not re.search(r'[A-Z]', password):
        errors.append('Password must contain at least one uppercase letter')
    if not re.search(r'[a-z]', password):
        errors.append('Password must contain at least one lowercase letter')
    if not re.search(r'[0-9]', password):
        errors.append('Password must contain at least one digit')
    return errors


@auth_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('applicant.dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        user_data = users_collection.find_one({'email': email})
        if user_data and check_password_hash(user_data['password_hash'], password):
            user = User(user_data)
            login_user(user)
            return redirect(url_for('applicant.dashboard'))

        flash('Invalid email or password', 'error')
        return redirect(url_for('auth.login'))

    return render_template('login.html')


@auth_bp.route('/signup', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('applicant.dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        password_errors = validate_password(password)
        if password_errors:
            for error in password_errors:
                flash(error, 'error')
            return redirect(url_for('auth.signup'))

        if users_collection.find_one({'email': email}):
            flash('Email already exists', 'error')
            return redirect(url_for('auth.signup'))

        user_data = {
            'email': email,
            'password_hash': generate_password_hash(password),
            'created_at': datetime.datetime.now(timezone.utc)
        }
        users_collection.insert_one(user_data)

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('signup.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


@auth_bp.route('/login_recruiter', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login_recruiter():
    if current_user.is_authenticated:
        return redirect(url_for('recruiter.recruiter_dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        recruiter_data = recruiters_collection.find_one({'email': email})
        if recruiter_data and check_password_hash(recruiter_data['password_hash'], password):
            recruiter = Recruiter(recruiter_data)
            login_user(recruiter)
            return redirect(url_for('recruiter.recruiter_dashboard'))

        flash('Invalid email or password', 'error')
        return redirect(url_for('auth.login_recruiter'))

    return render_template('recruiter_login.html')


@auth_bp.route('/signup_recruiter', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def signup_recruiter():
    if current_user.is_authenticated:
        return redirect(url_for('recruiter.recruiter_dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        name = request.form.get('name', '').strip()
        company = request.form.get('company', '').strip()

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('auth.signup_recruiter'))

        password_errors = validate_password(password)
        if password_errors:
            for error in password_errors:
                flash(error, 'error')
            return redirect(url_for('auth.signup_recruiter'))

        if recruiters_collection.find_one({'email': email}):
            flash('Email already exists', 'error')
            return redirect(url_for('auth.signup_recruiter'))

        recruiter_data = {
            'email': email,
            'password_hash': generate_password_hash(password),
            'name': name,
            'company': company,
            'created_at': datetime.datetime.now(timezone.utc),
            'updated_at': datetime.datetime.now(timezone.utc)
        }
        recruiters_collection.insert_one(recruiter_data)

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('auth.login_recruiter'))

    return render_template('recruiter_signup.html')
