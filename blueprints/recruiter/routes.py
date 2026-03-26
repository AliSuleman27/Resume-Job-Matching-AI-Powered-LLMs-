import os
import json
import logging
import datetime
from datetime import timezone
from collections import defaultdict
from flask import render_template, request, redirect, url_for, flash, current_app, jsonify, session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
from blueprints.recruiter import recruiter_bp
from blueprints.auth.user_models import Recruiter
import re
from extensions import (
    jobs_collection, applications_collection, resumes_collection, ai_results_collection, matcher,
    users_collection, talent_pool_collection
)
from services.llm_service import call_job_llm, extract_text_from_file, allowed_file
from services.email_service import dispatch_status_email, dispatch_interview_email, dispatch_cancellation_email
from services.pipeline_service import get_pipeline_stages, validate_stage_transition, get_notification_stages, build_pipeline_stages
from services.google_calendar_service import (
    google_calendar_configured, build_auth_url, exchange_code_for_tokens,
    store_google_tokens, recruiter_has_google, disconnect_google,
    create_calendar_event, delete_calendar_event,
)

logger = logging.getLogger(__name__)


@recruiter_bp.route('/dashboard')
@login_required
def recruiter_dashboard():
    if not isinstance(current_user, Recruiter):
        flash('Access denied', 'error')
        return redirect(url_for('applicant.landing'))

    jobs = list(jobs_collection.find({'recruiter_id': ObjectId(current_user.id)}))
    job_ids = [job['_id'] for job in jobs]

    total_applicants = applications_collection.count_documents({'job_id': {'$in': job_ids}}) if job_ids else 0

    # Compute real top candidates (score >= 0.7)
    top_candidates = 0
    ai_analyses_count = 0
    if job_ids:
        ai_results = list(ai_results_collection.find({'job_id': {'$in': job_ids}}))
        ai_analyses_count = len(ai_results)
        for result in ai_results:
            for candidate in result.get('ranked_candidates', []):
                if candidate.get('overall_score', 0) >= 0.7:
                    top_candidates += 1

    pool_count = talent_pool_collection.count_documents({'recruiter_id': ObjectId(current_user.id)})

    google_configured = google_calendar_configured()
    google_connected = recruiter_has_google(current_user.id) if google_configured else False

    # Fetch scheduled interviews
    upcoming_interviews = []
    past_interviews = []
    if job_ids:
        now = datetime.datetime.utcnow()
        interview_docs = list(applications_collection.aggregate([
            {'$match': {
                'job_id': {'$in': job_ids},
                'interview_datetime': {'$exists': True},
            }},
            {'$lookup': {
                'from': 'jobs', 'localField': 'job_id',
                'foreignField': '_id', 'as': 'job',
            }},
            {'$unwind': '$job'},
            {'$lookup': {
                'from': 'users', 'localField': 'user_id',
                'foreignField': '_id', 'as': 'user',
            }},
            {'$unwind': '$user'},
            {'$project': {
                'interview_datetime': 1,
                'interview_duration': 1,
                'interview_timezone': 1,
                'interview_notes': 1,
                'interview_meet_link': 1,
                'interview_event_id': 1,
                'status': 1,
                'job_title': '$job.parsed_data.title',
                'company': '$job.company',
                'applicant_name': '$user.name',
                'applicant_email': '$user.email',
            }},
            {'$sort': {'interview_datetime': 1}},
        ]))

        for doc in interview_docs:
            doc['_id_str'] = str(doc['_id'])
            if doc['interview_datetime'] >= now:
                upcoming_interviews.append(doc)
            else:
                past_interviews.append(doc)

        # Past interviews most recent first
        past_interviews.reverse()

    interview_count = len(upcoming_interviews) + len(past_interviews)

    return render_template('recruiter_dashboard.html',
                           jobs=jobs,
                           total_applicants=total_applicants,
                           top_candidates=top_candidates,
                           ai_analyses_count=ai_analyses_count,
                           pool_count=pool_count,
                           google_configured=google_configured,
                           google_connected=google_connected,
                           upcoming_interviews=upcoming_interviews,
                           past_interviews=past_interviews,
                           interview_count=interview_count)


@recruiter_bp.route('/create_job', methods=['GET'])
@login_required
def create_job():
    if not isinstance(current_user, Recruiter):
        flash('Access denied', 'error')
        return redirect(url_for('applicant.landing'))
    return render_template('create_job.html')


@recruiter_bp.route('/upload_job', methods=['POST'])
@login_required
def upload_job():
    """AI-parse an uploaded JD file and return parsed data for review."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        file_type = filename.rsplit('.', 1)[-1].lower()
        job_text = extract_text_from_file(temp_path, file_type)

        llm_response = call_job_llm(job_text)
        if not llm_response.get('output'):
            return jsonify({'error': 'Failed to parse job description'}), 500

        parsed_job = json.loads(llm_response['output'])

        try:
            os.remove(temp_path)
        except OSError:
            pass

        return jsonify({
            'success': True,
            'parsed_data': parsed_job,
            'original_filename': filename,
            'job_description': job_text
        })
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON for job description parsing")
        return jsonify({'error': 'Error parsing job description output. Please try again.'}), 500
    except Exception as e:
        logger.error(f"Error processing job file: {e}")
        return jsonify({'error': 'Error processing job description'}), 500


@recruiter_bp.route('/save_job', methods=['POST'])
@login_required
def save_job():
    """Save a new job (from either AI-parse review or manual form)."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    parsed_data = data.get('parsed_data')
    if not parsed_data:
        return jsonify({'error': 'parsed_data is required'}), 400

    # Ensure parsed_data has all fields the Pydantic model & templates expect
    import uuid as _uuid
    now_str = datetime.datetime.now(timezone.utc).strftime('%Y-%m-%d')
    parsed_data.setdefault('job_id', str(_uuid.uuid4()))
    parsed_data.setdefault('industry', '')
    parsed_data.setdefault('application_url', '')
    parsed_data.setdefault('department', '')
    parsed_data.setdefault('function', '')
    parsed_data.setdefault('description', '')
    parsed_data.setdefault('summary', '')
    parsed_data.setdefault('employment_type', 'full_time')
    parsed_data.setdefault('job_level', 'mid')
    parsed_data.setdefault('is_remote', False)
    parsed_data.setdefault('is_hybrid', False)
    parsed_data.setdefault('is_onsite', True)
    parsed_data.setdefault('posting_date', now_str)
    parsed_data.setdefault('closing_date', '')
    parsed_data.setdefault('salary', None)
    parsed_data.setdefault('benefits', [])
    parsed_data.setdefault('responsibilities', [])
    parsed_data.setdefault('requirements', [])
    parsed_data.setdefault('nice_to_have', [])
    parsed_data.setdefault('languages', [])

    # Nested objects — fill defaults so templates don't crash on attribute access
    if not parsed_data.get('company') or not isinstance(parsed_data['company'], dict):
        parsed_data['company'] = {'name': current_user.company or '', 'website': '', 'description': ''}
    else:
        parsed_data['company'].setdefault('name', current_user.company or '')
        parsed_data['company'].setdefault('website', '')
        parsed_data['company'].setdefault('description', '')

    if not parsed_data.get('skills') or not isinstance(parsed_data['skills'], dict):
        parsed_data['skills'] = {'mandatory': [], 'optional': [], 'tools': []}
    else:
        parsed_data['skills'].setdefault('mandatory', [])
        parsed_data['skills'].setdefault('optional', [])
        parsed_data['skills'].setdefault('tools', [])

    if not parsed_data.get('qualifications') or not isinstance(parsed_data['qualifications'], dict):
        parsed_data['qualifications'] = {
            'education': [{'degree': '', 'field_of_study': '', 'level': 'bachelor'}],
            'experience_years': {'min': 0, 'max': 5},
            'certifications': []
        }
    else:
        q = parsed_data['qualifications']
        q.setdefault('certifications', [])
        q.setdefault('experience_years', {'min': 0, 'max': 5})
        if q.get('education'):
            for edu in q['education']:
                edu.setdefault('level', 'bachelor')
                edu.setdefault('degree', '')
                edu.setdefault('field_of_study', '')
        else:
            q['education'] = [{'degree': '', 'field_of_study': '', 'level': 'bachelor'}]

    # Locations — ensure zip_code exists
    for loc in parsed_data.get('locations', []):
        loc.setdefault('zip_code', '')
        loc.setdefault('city', '')
        loc.setdefault('state', '')
        loc.setdefault('country', '')
        loc.setdefault('remote', False)
    if not parsed_data.get('locations'):
        parsed_data['locations'] = [{'city': '', 'state': '', 'country': '', 'zip_code': '', 'remote': False}]

    parsed_data.setdefault('analytics', {'views': 0, 'applications': 0})
    parsed_data.setdefault('metadata', {
        'created_at': now_str, 'updated_at': now_str,
        'created_by_user_id': str(current_user.id), 'source': 'manual'
    })

    # Build screening questions list (validate ids)
    screening_questions = data.get('screening_questions', [])
    for q in screening_questions:
        if not q.get('id') or not q.get('type') or not q.get('question'):
            return jsonify({'error': 'Each screening question needs id, type, and question text'}), 400

    # Build pipeline stages
    custom_stages = data.get('custom_pipeline_stages', [])
    pipeline_stages = build_pipeline_stages(custom_stages) if custom_stages else None

    # Notification stages
    notification_stages = data.get('notification_stages')

    creation_mode = data.get('creation_mode', 'manual')

    job_data = {
        'recruiter_id': ObjectId(current_user.id),
        'company': current_user.company,
        'original_filename': data.get('original_filename', ''),
        'job_description': data.get('job_description', ''),
        'parsed_data': parsed_data,
        'creation_mode': creation_mode,
        'created_at': datetime.datetime.now(timezone.utc),
        'updated_at': datetime.datetime.now(timezone.utc),
    }

    if screening_questions:
        job_data['screening_questions'] = screening_questions
    if pipeline_stages:
        job_data['pipeline_stages'] = pipeline_stages
    if notification_stages:
        job_data['notification_stages'] = notification_stages

    jobs_collection.insert_one(job_data)
    matcher.process_job(jsonJob=parsed_data)

    return jsonify({'success': True, 'job_id': str(job_data['_id'])})


@recruiter_bp.route('/job/<job_id>')
@login_required
def view_job(job_id):
    if not isinstance(current_user, Recruiter):
        flash('Access denied', 'error')
        return redirect(url_for('applicant.landing'))

    job = jobs_collection.find_one({
        '_id': ObjectId(job_id),
        'recruiter_id': ObjectId(current_user.id)
    })

    if not job:
        flash('Job not found', 'error')
        return redirect(url_for('recruiter.recruiter_dashboard'))

    return render_template('view_job.html', job=job)


@recruiter_bp.route('/applicants')
@login_required
def view_applicants():
    """View all applicants across all jobs posted by this recruiter"""
    if not isinstance(current_user, Recruiter):
        flash('Access denied', 'error')
        return redirect(url_for('applicant.landing'))

    recruiter_jobs = list(jobs_collection.find(
        {'recruiter_id': ObjectId(current_user.id)},
        {'_id': 1, 'parsed_data.title': 1}
    ))

    job_ids = [job['_id'] for job in recruiter_jobs]

    applications = list(applications_collection.aggregate([
        {'$match': {'job_id': {'$in': job_ids}}},
        {'$lookup': {'from': 'users', 'localField': 'user_id', 'foreignField': '_id', 'as': 'user'}},
        {'$unwind': '$user'},
        {'$lookup': {'from': 'parsed_resumes', 'localField': 'resume_id', 'foreignField': '_id', 'as': 'resume'}},
        {'$unwind': '$resume'},
        {'$lookup': {'from': 'jobs', 'localField': 'job_id', 'foreignField': '_id', 'as': 'job'}},
        {'$unwind': '$job'},
        {'$project': {
            'user_id': 1,
            'user_email': '$user.email',
            'job_title': '$job.parsed_data.title',
            'resume_data': '$resume',
            'status': 1,
            'applied_at': 1,
            'cover_message': 1
        }}
    ]))

    return render_template('recruiter/applicants.html',
                           applications=applications,
                           total_applicants=len(applications))


@recruiter_bp.route('/job/<job_id>/applicants')
@login_required
def view_job_applicants(job_id):
    """View applicants for a specific job with statistics"""
    if not isinstance(current_user, Recruiter):
        flash('Access denied', 'error')
        return redirect(url_for('applicant.landing'))

    job = jobs_collection.find_one({
        '_id': ObjectId(job_id),
        'recruiter_id': ObjectId(current_user.id)
    })

    if not job:
        flash('Job not found', 'error')
        return redirect(url_for('recruiter.recruiter_dashboard'))

    applications = list(applications_collection.aggregate([
        {'$match': {'job_id': ObjectId(job_id)}},
        {'$lookup': {'from': 'users', 'localField': 'user_id', 'foreignField': '_id', 'as': 'user'}},
        {'$unwind': '$user'},
        {'$lookup': {'from': 'parsed_resumes', 'localField': 'resume_id', 'foreignField': '_id', 'as': 'resume'}},
        {'$unwind': '$resume'},
        {'$project': {
            '_id': 1,
            'resume_id': 1,
            'user_id': 1,
            'user_email': '$user.email',
            'resume_data': '$resume.parsed_data',
            'status': 1,
            'applied_at': 1,
            'cover_message': 1,
            'feedback': 1,
            'feedback_at': 1
        }}
    ]))

    status_counts = defaultdict(int)
    skills = defaultdict(int)
    experience_levels = defaultdict(int)
    total_applicants = len(applications)

    for app in applications:
        status_counts[app['status']] += 1

        resume_data = app.get('resume_data', {})
        for skill in resume_data.get('skills', []):
            skills[skill['skill_name']] += 1

        total_exp = 0
        for exp in resume_data.get('experience', []):
            if exp.get('start_date') and exp.get('end_date'):
                try:
                    from datetime import datetime as dt
                    start = dt.strptime(exp['start_date'], "%Y-%m-%d")
                    end = dt.strptime(exp['end_date'], "%Y-%m-%d") if exp['end_date'] else dt.now(timezone.utc)
                    total_exp += (end - start).days / 365.25
                except Exception:
                    continue

        exp_level = 'Junior' if total_exp < 3 else 'Mid-level' if total_exp < 7 else 'Senior'
        experience_levels[exp_level] += 1

    top_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)[:10]

    pipeline_stages = get_pipeline_stages(job)

    return render_template('recruiter/job_applicants.html',
                           job=job,
                           applications=applications,
                           total_applicants=total_applicants,
                           status_counts=status_counts,
                           top_skills=top_skills,
                           experience_levels=experience_levels,
                           pipeline_stages=pipeline_stages)


@recruiter_bp.route('/view_resume/<resume_id>')
@login_required
def recruiter_view_resume(resume_id):
    if not isinstance(current_user, Recruiter):
        flash('Access denied', 'error')
        return redirect(url_for('applicant.landing'))

    resume = resumes_collection.find_one({'_id': ObjectId(resume_id)})

    if not resume:
        flash('Resume not found', 'error')
        return redirect(url_for('recruiter.recruiter_dashboard'))

    application = applications_collection.find_one({
        'resume_id': ObjectId(resume_id),
        'job_id': {'$in': jobs_collection.find(
            {'recruiter_id': ObjectId(current_user.id)},
            {'_id': 1}
        ).distinct('_id')}
    })

    if not application:
        flash('Access to this resume is restricted', 'error')
        return redirect(url_for('recruiter.recruiter_dashboard'))

    return render_template('recruiter/resume_view.html', resume=resume)


@recruiter_bp.route('/api/applications/<application_id>/status', methods=['PUT'])
@login_required
def update_application_status(application_id):
    """Update a single application's status"""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    data = request.get_json()
    new_status = data.get('status')

    # Verify the application belongs to one of the recruiter's jobs
    application = applications_collection.find_one({'_id': ObjectId(application_id)})
    if not application:
        return jsonify({'error': 'Application not found'}), 404

    job = jobs_collection.find_one({
        '_id': application['job_id'],
        'recruiter_id': ObjectId(current_user.id)
    })
    if not job:
        return jsonify({'error': 'Access denied'}), 403

    # Validate against this job's pipeline stages (dynamic, not hardcoded)
    valid_statuses = get_pipeline_stages(job)
    if not validate_stage_transition(application.get('status'), new_status, valid_statuses):
        return jsonify({'error': 'Invalid status'}), 400

    update_fields = {'status': new_status, 'updated_at': datetime.datetime.now(timezone.utc)}

    feedback = data.get('feedback')
    if feedback is not None:
        feedback = str(feedback)[:2000]
        update_fields['feedback'] = feedback
        update_fields['feedback_at'] = datetime.datetime.now(timezone.utc)

    applications_collection.update_one(
        {'_id': ObjectId(application_id)},
        {'$set': update_fields}
    )

    # Email notification (non-blocking) — only for notification-enabled stages
    notified = False
    notify_stages = get_notification_stages(job)
    if data.get('notify_applicant', True) and new_status in notify_stages:
        try:
            dispatch_status_email(application_id, new_status, feedback)
            notified = True
        except Exception as e:
            logger.error(f"Email dispatch failed for {application_id}: {e}")

    return jsonify({'success': True, 'status': new_status, 'notified': notified})


@recruiter_bp.route('/api/applications/<application_id>/feedback', methods=['PUT'])
@login_required
def update_application_feedback(application_id):
    """Update feedback on an application without changing status"""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    data = request.get_json()
    feedback = data.get('feedback', '')
    if not feedback or not feedback.strip():
        return jsonify({'error': 'Feedback text is required'}), 400

    feedback = str(feedback)[:2000]

    application = applications_collection.find_one({'_id': ObjectId(application_id)})
    if not application:
        return jsonify({'error': 'Application not found'}), 404

    job = jobs_collection.find_one({
        '_id': application['job_id'],
        'recruiter_id': ObjectId(current_user.id)
    })
    if not job:
        return jsonify({'error': 'Access denied'}), 403

    applications_collection.update_one(
        {'_id': ObjectId(application_id)},
        {'$set': {
            'feedback': feedback,
            'feedback_at': datetime.datetime.now(timezone.utc),
            'updated_at': datetime.datetime.now(timezone.utc)
        }}
    )

    return jsonify({'success': True})


@recruiter_bp.route('/api/applications/<application_id>/contact', methods=['GET'])
@login_required
def get_application_contact(application_id):
    """Get applicant contact info for the email modal"""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    application = applications_collection.find_one({'_id': ObjectId(application_id)})
    if not application:
        return jsonify({'error': 'Application not found'}), 404

    job = jobs_collection.find_one({
        '_id': application['job_id'],
        'recruiter_id': ObjectId(current_user.id)
    })
    if not job:
        return jsonify({'error': 'Access denied'}), 403

    user = users_collection.find_one({'_id': application['user_id']})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({
        'success': True,
        'email': user.get('email', ''),
        'job_title': job.get('parsed_data', {}).get('title', ''),
        'company': job.get('company', '')
    })


@recruiter_bp.route('/api/applications/bulk-status', methods=['PUT'])
@login_required
def bulk_update_status():
    """Bulk update application statuses"""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    data = request.get_json()
    application_ids = data.get('application_ids', [])
    new_status = data.get('status')
    job_id = data.get('job_id')

    # Validate against pipeline stages if job_id provided, else use defaults
    if job_id:
        job = jobs_collection.find_one({'_id': ObjectId(job_id), 'recruiter_id': ObjectId(current_user.id)})
        valid_statuses = get_pipeline_stages(job) if job else get_pipeline_stages({})
    else:
        valid_statuses = get_pipeline_stages({})

    if new_status not in valid_statuses:
        return jsonify({'error': 'Invalid status'}), 400

    if not application_ids:
        return jsonify({'error': 'No applications selected'}), 400

    # Get recruiter's job IDs
    recruiter_job_ids = jobs_collection.find(
        {'recruiter_id': ObjectId(current_user.id)}, {'_id': 1}
    ).distinct('_id')

    obj_ids = [ObjectId(aid) for aid in application_ids]

    result = applications_collection.update_many(
        {
            '_id': {'$in': obj_ids},
            'job_id': {'$in': recruiter_job_ids}
        },
        {'$set': {'status': new_status, 'updated_at': datetime.datetime.now(timezone.utc)}}
    )

    # Email notifications (non-blocking)
    if data.get('notify_applicant', True) and new_status != 'submitted':
        for aid in application_ids:
            try:
                dispatch_status_email(aid, new_status)
            except Exception as e:
                logger.error(f"Email dispatch failed for {aid}: {e}")

    return jsonify({'success': True, 'updated': result.modified_count})


@recruiter_bp.route('/candidate/<resume_id>/details')
@login_required
def candidate_details(resume_id):
    """Get candidate resume details for modal view — authorization enforced"""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    resume = resumes_collection.find_one({'_id': ObjectId(resume_id)})
    if not resume:
        return jsonify({'success': False, 'error': 'Resume not found'}), 404

    # IDOR FIX: verify this resume belongs to an applicant for one of the recruiter's jobs
    recruiter_job_ids = jobs_collection.find(
        {'recruiter_id': ObjectId(current_user.id)}, {'_id': 1}
    ).distinct('_id')

    application = applications_collection.find_one({
        'resume_id': ObjectId(resume_id),
        'job_id': {'$in': recruiter_job_ids}
    })

    if not application:
        return jsonify({'error': 'Access denied'}), 403

    return jsonify({'success': True, 'candidate': resume.get('parsed_data', {})})


# ---------------------------------------------------------------------------
# Google Calendar OAuth routes
# ---------------------------------------------------------------------------

@recruiter_bp.route('/google/connect')
@login_required
def google_connect():
    """Redirect to Google OAuth consent screen."""
    if not isinstance(current_user, Recruiter):
        flash('Access denied', 'error')
        return redirect(url_for('applicant.landing'))

    if not google_calendar_configured():
        flash('Google Calendar integration is not configured', 'error')
        return redirect(url_for('recruiter.recruiter_dashboard'))

    auth_url, code_verifier = build_auth_url()
    session['google_code_verifier'] = code_verifier
    return redirect(auth_url)


@recruiter_bp.route('/google/callback')
@login_required
def google_callback():
    """Handle OAuth callback — exchange code for tokens."""
    if not isinstance(current_user, Recruiter):
        flash('Access denied', 'error')
        return redirect(url_for('applicant.landing'))

    try:
        code_verifier = session.pop('google_code_verifier', None)
        token_data = exchange_code_for_tokens(request.url, code_verifier)
        store_google_tokens(current_user.id, token_data)
        flash('Google Calendar connected successfully!', 'success')
    except Exception as e:
        logger.error(f"Google OAuth callback failed: {e}")
        flash('Failed to connect Google Calendar. Please try again.', 'error')

    return redirect(url_for('recruiter.recruiter_dashboard'))


@recruiter_bp.route('/google/disconnect', methods=['POST'])
@login_required
def google_disconnect():
    """Remove Google tokens from recruiter document."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    disconnect_google(current_user.id)
    return jsonify({'success': True})


@recruiter_bp.route('/google/status')
@login_required
def google_status():
    """Return JSON with Google Calendar connection status."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    configured = google_calendar_configured()
    connected = recruiter_has_google(current_user.id) if configured else False
    return jsonify({'configured': configured, 'connected': connected})


# ---------------------------------------------------------------------------
# Interview scheduling
# ---------------------------------------------------------------------------

@recruiter_bp.route('/api/applications/<application_id>/schedule-interview', methods=['POST'])
@login_required
def schedule_interview(application_id):
    """Create a calendar event, store interview data, dispatch email."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    data = request.get_json()
    if not data or not data.get('datetime'):
        return jsonify({'error': 'Interview datetime is required'}), 400

    # Verify ownership
    application = applications_collection.find_one({'_id': ObjectId(application_id)})
    if not application:
        return jsonify({'error': 'Application not found'}), 404

    job = jobs_collection.find_one({
        '_id': application['job_id'],
        'recruiter_id': ObjectId(current_user.id),
    })
    if not job:
        return jsonify({'error': 'Access denied'}), 403

    # Parse interview details
    try:
        interview_dt = datetime.datetime.fromisoformat(data['datetime'])
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid datetime format'}), 400

    duration = int(data.get('duration', 30))
    notes = data.get('notes', '')
    tz = data.get('timezone', 'UTC')

    job_title = job.get('parsed_data', {}).get('title', 'Interview')
    company = job.get('company', '')

    # Get applicant email for calendar invite
    user = users_collection.find_one({'_id': application['user_id']})
    applicant_email = user.get('email', '') if user else ''
    applicant_name = user.get('name', applicant_email.split('@')[0]) if user else 'Candidate'

    meet_link = ''
    event_id = ''

    # Try Google Calendar if connected
    if google_calendar_configured() and recruiter_has_google(current_user.id):
        attendees = [applicant_email] if applicant_email else []
        summary = f"Interview: {applicant_name} — {job_title} ({company})"
        description = f"Interview for {job_title} at {company}"
        if notes:
            description += f"\n\nNotes: {notes}"

        result = create_calendar_event(
            current_user.id, summary, description,
            interview_dt, duration, tz, attendees,
        )
        if result.get('error'):
            logger.warning(f"Calendar event creation failed: {result['error']}")
        else:
            meet_link = result.get('meet_link', '')
            event_id = result.get('event_id', '')

    # Store interview data on application document
    update_fields = {
        'status': 'shortlisted',
        'interview_datetime': interview_dt,
        'interview_duration': duration,
        'interview_timezone': tz,
        'interview_notes': notes,
        'interview_meet_link': meet_link,
        'interview_event_id': event_id,
        'interview_scheduled_at': datetime.datetime.now(datetime.timezone.utc),
        'interview_scheduled_by': ObjectId(current_user.id),
        'updated_at': datetime.datetime.now(datetime.timezone.utc),
    }
    applications_collection.update_one(
        {'_id': ObjectId(application_id)},
        {'$set': update_fields},
    )

    # Dispatch interview email (non-blocking)
    try:
        dispatch_interview_email(application_id, interview_dt, duration, meet_link, notes)
    except Exception as e:
        logger.error(f"Interview email dispatch failed for {application_id}: {e}")

    return jsonify({
        'success': True,
        'meet_link': meet_link,
        'event_id': event_id,
    })


@recruiter_bp.route('/api/applications/<application_id>/cancel-interview', methods=['POST'])
@login_required
def cancel_interview(application_id):
    """Cancel a scheduled interview — remove calendar event, unset fields, notify candidate."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    application = applications_collection.find_one({'_id': ObjectId(application_id)})
    if not application:
        return jsonify({'error': 'Application not found'}), 404

    job = jobs_collection.find_one({
        '_id': application['job_id'],
        'recruiter_id': ObjectId(current_user.id),
    })
    if not job:
        return jsonify({'error': 'Access denied'}), 403

    if not application.get('interview_datetime'):
        return jsonify({'error': 'No interview scheduled for this application'}), 400

    interview_datetime = application['interview_datetime']
    event_id = application.get('interview_event_id', '')

    # Delete Google Calendar event if exists
    calendar_deleted = False
    if event_id and google_calendar_configured() and recruiter_has_google(current_user.id):
        result = delete_calendar_event(current_user.id, event_id)
        if result.get('success'):
            calendar_deleted = True
        else:
            logger.warning(f"Failed to delete calendar event for {application_id}: {result.get('error')}")

    # Unset all interview fields and revert status
    interview_fields = [
        'interview_datetime', 'interview_duration', 'interview_timezone',
        'interview_notes', 'interview_meet_link', 'interview_event_id',
        'interview_scheduled_at', 'interview_scheduled_by',
    ]
    unset_dict = {field: '' for field in interview_fields}

    new_status = 'shortlisted' if application.get('status') == 'shortlisted' else 'applied'
    applications_collection.update_one(
        {'_id': ObjectId(application_id)},
        {
            '$unset': unset_dict,
            '$set': {
                'status': new_status,
                'updated_at': datetime.datetime.now(timezone.utc),
            },
        },
    )

    # Send cancellation email (non-blocking)
    try:
        dispatch_cancellation_email(application_id, interview_datetime)
    except Exception as e:
        logger.error(f"Cancellation email dispatch failed for {application_id}: {e}")

    return jsonify({'success': True, 'calendar_deleted': calendar_deleted})


# ---------------------------------------------------------------------------
# Talent Pool / CRM
# ---------------------------------------------------------------------------

@recruiter_bp.route('/talent-pool')
@login_required
def talent_pool():
    """Main talent pool page with search/filter."""
    if not isinstance(current_user, Recruiter):
        flash('Access denied', 'error')
        return redirect(url_for('applicant.landing'))

    recruiter_id = ObjectId(current_user.id)
    query = {'recruiter_id': recruiter_id}

    # Search filter
    search = request.args.get('q', '').strip()
    if search:
        search_regex = {'$regex': re.escape(search), '$options': 'i'}
        query['$or'] = [
            {'candidate_name': search_regex},
            {'candidate_title': search_regex},
            {'candidate_email': search_regex},
            {'candidate_skills': search_regex},
            {'tags': search_regex},
        ]

    # Tag filter
    tag = request.args.get('tag', '').strip().lower()
    if tag:
        query['tags'] = tag

    # Star rating filter
    min_rating = request.args.get('min_rating', '', type=str)
    if min_rating and min_rating.isdigit():
        query['star_rating'] = {'$gte': int(min_rating)}

    # Source job filter
    source_job_id = request.args.get('source_job', '').strip()
    if source_job_id:
        try:
            query['source_job_id'] = ObjectId(source_job_id)
        except Exception:
            pass

    entries = list(talent_pool_collection.find(query).sort('added_at', -1))

    # Recruiter's jobs for the source filter dropdown
    recruiter_jobs = list(jobs_collection.find(
        {'recruiter_id': recruiter_id}, {'_id': 1, 'parsed_data.title': 1}
    ))

    # Distinct tags for filter
    all_tags = talent_pool_collection.distinct('tags', {'recruiter_id': recruiter_id})

    return render_template('recruiter/talent_pool.html',
                           entries=entries,
                           recruiter_jobs=recruiter_jobs,
                           all_tags=sorted(all_tags),
                           search=search,
                           active_tag=tag,
                           active_rating=min_rating,
                           active_source=source_job_id)


@recruiter_bp.route('/api/talent-pool/add', methods=['POST'])
@login_required
def talent_pool_add():
    """Save a candidate to the talent pool."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    data = request.get_json()
    resume_id = data.get('resume_id')
    job_id = data.get('job_id')
    if not resume_id or not job_id:
        return jsonify({'error': 'resume_id and job_id are required'}), 400

    recruiter_id = ObjectId(current_user.id)

    # Check duplicate
    if talent_pool_collection.find_one({'recruiter_id': recruiter_id, 'resume_id': ObjectId(resume_id)}):
        return jsonify({'error': 'Candidate already in your talent pool'}), 409

    # Verify job belongs to recruiter
    job = jobs_collection.find_one({'_id': ObjectId(job_id), 'recruiter_id': recruiter_id})
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    # Get resume data
    resume = resumes_collection.find_one({'_id': ObjectId(resume_id)})
    if not resume:
        return jsonify({'error': 'Resume not found'}), 404

    parsed = resume.get('parsed_data', {})
    basic = parsed.get('basic_info', {})
    skills_list = [s.get('skill_name', s) if isinstance(s, dict) else str(s) for s in parsed.get('skills', [])]

    # Get user_id from application
    application = applications_collection.find_one({
        'resume_id': ObjectId(resume_id),
        'job_id': ObjectId(job_id)
    })

    # Get AI score if available
    source_score = None
    if data.get('score') is not None:
        try:
            source_score = round(float(data['score']), 4)
        except (ValueError, TypeError):
            pass

    now = datetime.datetime.now(timezone.utc)
    tags = [t.strip().lower() for t in data.get('tags', []) if t.strip()][:20]
    notes = str(data.get('notes', ''))[:2000]
    star_rating = data.get('star_rating')
    if star_rating is not None:
        star_rating = max(1, min(5, int(star_rating)))

    entry = {
        'recruiter_id': recruiter_id,
        'resume_id': ObjectId(resume_id),
        'user_id': application['user_id'] if application else None,
        'source_job_id': ObjectId(job_id),
        'source_job_title': job.get('parsed_data', {}).get('title', ''),
        'source_application_id': application['_id'] if application else None,
        'source_score': source_score,
        'tags': tags,
        'notes': notes,
        'star_rating': star_rating,
        'candidate_name': basic.get('full_name', 'Unknown'),
        'candidate_title': basic.get('title', ''),
        'candidate_email': basic.get('email', ''),
        'candidate_skills': skills_list,
        'added_at': now,
        'updated_at': now,
    }

    result = talent_pool_collection.insert_one(entry)
    return jsonify({'success': True, 'entry_id': str(result.inserted_id)})


@recruiter_bp.route('/api/talent-pool/<entry_id>', methods=['PUT'])
@login_required
def talent_pool_update(entry_id):
    """Update tags/notes/rating on a talent pool entry."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    entry = talent_pool_collection.find_one({
        '_id': ObjectId(entry_id),
        'recruiter_id': ObjectId(current_user.id)
    })
    if not entry:
        return jsonify({'error': 'Entry not found'}), 404

    data = request.get_json()
    update = {'updated_at': datetime.datetime.now(timezone.utc)}

    if 'tags' in data:
        update['tags'] = [t.strip().lower() for t in data['tags'] if t.strip()][:20]
    if 'notes' in data:
        update['notes'] = str(data['notes'])[:2000]
    if 'star_rating' in data:
        sr = data['star_rating']
        update['star_rating'] = max(1, min(5, int(sr))) if sr else None

    talent_pool_collection.update_one({'_id': ObjectId(entry_id)}, {'$set': update})
    return jsonify({'success': True})


@recruiter_bp.route('/api/talent-pool/<entry_id>', methods=['DELETE'])
@login_required
def talent_pool_delete(entry_id):
    """Remove a candidate from the talent pool."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    result = talent_pool_collection.delete_one({
        '_id': ObjectId(entry_id),
        'recruiter_id': ObjectId(current_user.id)
    })
    if result.deleted_count == 0:
        return jsonify({'error': 'Entry not found'}), 404

    return jsonify({'success': True})


@recruiter_bp.route('/api/talent-pool/tags')
@login_required
def talent_pool_tags():
    """Distinct tags for autocomplete."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    tags = talent_pool_collection.distinct('tags', {'recruiter_id': ObjectId(current_user.id)})
    return jsonify({'tags': sorted(tags)})


@recruiter_bp.route('/api/talent-pool/match/<job_id>')
@login_required
def talent_pool_match(job_id):
    """Find talent pool candidates matching a job's skills."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    recruiter_id = ObjectId(current_user.id)
    job = jobs_collection.find_one({'_id': ObjectId(job_id), 'recruiter_id': recruiter_id})
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    # Extract skills from job parsed_data
    parsed = job.get('parsed_data', {})
    skills_section = parsed.get('skills', {})
    job_skills = []
    if isinstance(skills_section, dict):
        for key in ('mandatory', 'optional', 'tools'):
            job_skills.extend(skills_section.get(key, []))
    elif isinstance(skills_section, list):
        job_skills = skills_section

    if not job_skills:
        return jsonify({'count': 0, 'candidates': []})

    # Build regex OR for all job skills
    patterns = [re.escape(str(s)) for s in job_skills if s]
    if not patterns:
        return jsonify({'count': 0, 'candidates': []})

    regex = '|'.join(patterns)
    pool_entries = list(talent_pool_collection.find({
        'recruiter_id': recruiter_id,
        'candidate_skills': {'$regex': regex, '$options': 'i'}
    }))

    # Score by overlap count
    results = []
    for entry in pool_entries:
        cand_skills_lower = [s.lower() for s in entry.get('candidate_skills', [])]
        overlap = sum(1 for s in job_skills if str(s).lower() in ' '.join(cand_skills_lower))
        results.append({
            'entry_id': str(entry['_id']),
            'candidate_name': entry.get('candidate_name', ''),
            'candidate_title': entry.get('candidate_title', ''),
            'candidate_skills': entry.get('candidate_skills', []),
            'source_job_title': entry.get('source_job_title', ''),
            'star_rating': entry.get('star_rating'),
            'overlap': overlap,
        })

    results.sort(key=lambda x: x['overlap'], reverse=True)
    return jsonify({'count': len(results), 'candidates': results})
