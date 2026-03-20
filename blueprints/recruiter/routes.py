import os
import json
import logging
import datetime
from datetime import timezone
from collections import defaultdict
from flask import render_template, request, redirect, url_for, flash, current_app, jsonify
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
from blueprints.recruiter import recruiter_bp
from blueprints.auth.user_models import Recruiter
from extensions import (
    jobs_collection, applications_collection, resumes_collection, ai_results_collection, matcher,
    users_collection
)
from services.llm_service import call_job_llm, extract_text_from_file, allowed_file

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

    return render_template('recruiter_dashboard.html',
                           jobs=jobs,
                           total_applicants=total_applicants,
                           top_candidates=top_candidates,
                           ai_analyses_count=ai_analyses_count)


@recruiter_bp.route('/create_job', methods=['GET', 'POST'])
@login_required
def create_job():
    if not isinstance(current_user, Recruiter):
        flash('Access denied', 'error')
        return redirect(url_for('applicant.landing'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(temp_path)

                file_type = filename.rsplit('.', 1)[-1].lower()
                job_text = extract_text_from_file(temp_path, file_type)

                llm_response = call_job_llm(job_text)

                if not llm_response.get('output'):
                    flash('Failed to parse job description', 'error')
                    return redirect(request.url)

                parsed_job = json.loads(llm_response['output'])

                job_data = {
                    'recruiter_id': ObjectId(current_user.id),
                    'company': current_user.company,
                    'original_filename': filename,
                    'job_description': job_text,
                    'parsed_data': parsed_job,
                    'created_at': datetime.datetime.now(timezone.utc),
                    'updated_at': datetime.datetime.now(timezone.utc)
                }
                jobs_collection.insert_one(job_data)
                matcher.process_job(jsonJob=parsed_job)
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                flash('Job created successfully!', 'success')

                return redirect(url_for('recruiter.view_job', job_id=str(job_data['_id'])))

            except json.JSONDecodeError:
                logger.error("LLM returned invalid JSON for job description parsing")
                flash('Error parsing job description output. Please try again.', 'error')
                return redirect(request.url)
            except Exception as e:
                logger.error(f"Error processing job file: {e}")
                flash('Error processing job description', 'error')
                return redirect(request.url)
        else:
            flash('File type not allowed', 'error')
            return redirect(request.url)

    return render_template('create_job.html')


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

    return render_template('recruiter/job_applicants.html',
                           job=job,
                           applications=applications,
                           total_applicants=total_applicants,
                           status_counts=status_counts,
                           top_skills=top_skills,
                           experience_levels=experience_levels)


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
    valid_statuses = ['submitted', 'reviewed', 'shortlisted', 'rejected', 'hired']

    if new_status not in valid_statuses:
        return jsonify({'error': 'Invalid status'}), 400

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

    return jsonify({'success': True, 'status': new_status})


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
    valid_statuses = ['submitted', 'reviewed', 'shortlisted', 'rejected', 'hired']

    if new_status not in valid_statuses:
        return jsonify({'error': 'Invalid status'}), 400

    if not application_ids:
        return jsonify({'error': 'No applications selected'}), 400

    # Get recruiter's job IDs
    recruiter_job_ids = jobs_collection.find(
        {'recruiter_id': ObjectId(current_user.id)}, {'_id': 1}
    ).distinct('_id')

    result = applications_collection.update_many(
        {
            '_id': {'$in': [ObjectId(aid) for aid in application_ids]},
            'job_id': {'$in': recruiter_job_ids}
        },
        {'$set': {'status': new_status, 'updated_at': datetime.datetime.now(timezone.utc)}}
    )

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
