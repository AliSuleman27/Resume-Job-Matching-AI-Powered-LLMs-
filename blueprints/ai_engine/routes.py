import datetime
import os
import uuid
from datetime import timezone
import logging
import requests as http_requests
from flask import render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user
from bson.objectid import ObjectId
from blueprints.ai_engine import ai_engine_bp
from blueprints.auth.user_models import Recruiter
from extensions import (
    jobs_collection, ai_results_collection, resumes_collection,
    applications_collection, task_runner, background_tasks_collection,
    insight_service, chat_service, candidate_insights_collection
)

CF_MATCHING_URL = os.environ.get('CF_MATCHING_URL', '')
CF_AUTH_TOKEN = os.environ.get('CF_AUTH_TOKEN', '')

logger = logging.getLogger(__name__)


@ai_engine_bp.route('/job/<job_id>/run_ai_engine', methods=['POST'])
@login_required
def run_ai_engine(job_id):
    """Fire-and-forget: dispatch matching to Cloud Function, return task_id."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    job = jobs_collection.find_one({
        '_id': ObjectId(job_id),
        'recruiter_id': ObjectId(current_user.id)
    })

    if not job:
        return jsonify({'error': 'Job not found'}), 404

    try:
        existing_results = ai_results_collection.find_one({'job_id': ObjectId(job_id)})
        if existing_results:
            rerun_confirmed = request.form.get('confirm_rerun') or (request.get_json(silent=True) or {}).get('confirm_rerun', 'false')
            if rerun_confirmed not in ('true', True):
                return jsonify({'error': 'rerun_required', 'job_id': job_id}), 409

        task_id = str(uuid.uuid4())

        # Insert task doc into MongoDB (replaces in-memory TaskRunner for matching)
        background_tasks_collection.insert_one({
            'task_id': task_id,
            'job_id': str(job_id),
            'recruiter_id': str(current_user.id),
            'status': 'running',
            'created_at': datetime.datetime.now(timezone.utc),
            'updated_at': datetime.datetime.now(timezone.utc),
        })

        # Fire-and-forget POST to Cloud Function
        if CF_MATCHING_URL:
            try:
                http_requests.post(
                    CF_MATCHING_URL,
                    json={'task_id': task_id, 'job_id': str(job_id), 'recruiter_id': str(current_user.id)},
                    headers={'Authorization': f'Bearer {CF_AUTH_TOKEN}', 'Content-Type': 'application/json'},
                    timeout=5,
                )
            except http_requests.exceptions.Timeout:
                pass  # Expected — CF is still running
            except Exception as e:
                logger.warning(f"CF dispatch warning (task will still run): {e}")
        else:
            logger.warning("CF_MATCHING_URL not set — matching will not run")

        logger.info(f"AI engine task {task_id} dispatched for job {job_id}")
        return jsonify({'task_id': task_id, 'status': 'running'})

    except Exception as e:
        logger.error(f"Error starting AI engine for job {job_id}: {e}")
        return jsonify({'error': 'Failed to start AI analysis'}), 500


@ai_engine_bp.route('/job/<job_id>/matching_progress')
@login_required
def ai_matching_progress(job_id):
    """Render the matching progress page with spinner + polling."""
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

    task_id = request.args.get('task_id', '')
    return render_template('recruiter/matching_progress.html', job=job, task_id=task_id, job_id=job_id)


@ai_engine_bp.route('/job/<job_id>/view_ai_results')
@login_required
def view_ai_results(job_id):
    """View AI analysis results for a specific job"""
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

    ai_results = ai_results_collection.find_one({
        'job_id': ObjectId(job_id)
    })

    if not ai_results:
        flash('AI analysis has not been performed for this job yet. Please run the AI engine first.', 'error')
        return redirect(url_for('recruiter.recruiter_dashboard'))

    try:
        return render_template('recruiter/ranked_candidates.html',
                               job=job,
                               candidates=ai_results['ranked_candidates'],
                               stats=ai_results['statistics'],
                               analysis_date=ai_results['updated_at'])

    except Exception as e:
        logger.error(f"Error displaying AI results for job {job_id}: {e}")
        flash('Error loading AI results. Please try again.', 'error')
        return redirect(url_for('recruiter.recruiter_dashboard'))


@ai_engine_bp.route('/job/<job_id>/check_ai_status')
@login_required
def check_ai_status(job_id):
    """API endpoint to check if AI results exist for a job"""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    job = jobs_collection.find_one({
        '_id': ObjectId(job_id),
        'recruiter_id': ObjectId(current_user.id)
    })

    if not job:
        return jsonify({'error': 'Job not found'}), 404

    ai_results = ai_results_collection.find_one({
        'job_id': ObjectId(job_id)
    })

    return jsonify({
        'has_results': ai_results is not None,
        'last_updated': ai_results['updated_at'].isoformat() if ai_results else None
    })


@ai_engine_bp.route('/candidate/<candidate_id>/details')
@login_required
def get_candidate_details(candidate_id):
    """API endpoint to get detailed candidate information"""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    try:
        resume = resumes_collection.find_one({'_id': ObjectId(candidate_id)})
        if not resume:
            return jsonify({'error': 'Candidate not found'}), 404

        # Authorization: verify the candidate applied to one of this recruiter's jobs
        recruiter_job_ids = [job['_id'] for job in jobs_collection.find(
            {'recruiter_id': ObjectId(current_user.id)},
            {'_id': 1}
        )]

        application = applications_collection.find_one({
            'resume_id': ObjectId(candidate_id),
            'job_id': {'$in': recruiter_job_ids}
        })

        if not application:
            return jsonify({'error': 'Access denied'}), 403

        return jsonify({
            'success': True,
            'candidate': resume['parsed_data']
        })

    except Exception as e:
        logger.error(f"Error getting candidate details: {e}")
        return jsonify({'error': 'Internal server error'}), 500


# ------------------------------------------------------------------ #
#  CANDIDATE INSIGHT & COMPARISON ROUTES                              #
# ------------------------------------------------------------------ #

@ai_engine_bp.route('/job/<job_id>/candidate/<resume_id>/insight')
@login_required
def candidate_insight(job_id, resume_id):
    """Main comparison page for a candidate vs job description."""
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

    resume = resumes_collection.find_one({'_id': ObjectId(resume_id)})
    if not resume:
        flash('Candidate not found', 'error')
        return redirect(url_for('ai_engine.view_ai_results', job_id=job_id))

    try:
        insight = insight_service.get_or_create_insight(
            job_id, resume_id,
            resume['parsed_data'], job['parsed_data']
        )
        return render_template(
            'recruiter/candidate_comparison.html',
            job=job, resume=resume, insight=insight,
            job_id=job_id, resume_id=resume_id
        )
    except Exception as e:
        logger.error(f"Error loading insight for {resume_id}: {e}")
        flash('Error loading candidate insight. Please try again.', 'error')
        return redirect(url_for('ai_engine.view_ai_results', job_id=job_id))


@ai_engine_bp.route('/job/<job_id>/candidate/<resume_id>/generate-insights', methods=['POST'])
@login_required
def generate_insights(job_id, resume_id):
    """Lazy-load LLM insights via background task."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    job = jobs_collection.find_one({
        '_id': ObjectId(job_id),
        'recruiter_id': ObjectId(current_user.id)
    })
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    resume = resumes_collection.find_one({'_id': ObjectId(resume_id)})
    if not resume:
        return jsonify({'error': 'Candidate not found'}), 404

    # Check if already generated
    cached = candidate_insights_collection.find_one({
        'job_id': job_id, 'resume_id': resume_id
    })
    if cached and cached.get('llm_insights'):
        return jsonify({'status': 'completed', 'result': cached['llm_insights']})

    from models.resume_model import Resume as ResumeModel
    from models.job_description_model import JobDescription

    def _generate():
        r = ResumeModel(**resume['parsed_data'])
        j = JobDescription(**job['parsed_data'])
        insight_doc = candidate_insights_collection.find_one({
            'job_id': job_id, 'resume_id': resume_id
        })
        return insight_service.generate_insights(
            job_id, resume_id,
            insight_doc['skills_diff'], insight_doc['experience_diff'],
            insight_doc['education_diff'], r, j
        )

    bg_task_id = task_runner.submit(_generate)
    return jsonify({'status': 'running', 'task_id': bg_task_id})


@ai_engine_bp.route('/job/<job_id>/candidate/<resume_id>/generate-questions', methods=['POST'])
@login_required
def generate_questions(job_id, resume_id):
    """Lazy-load LLM interview questions via background task."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    job = jobs_collection.find_one({
        '_id': ObjectId(job_id),
        'recruiter_id': ObjectId(current_user.id)
    })
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    resume = resumes_collection.find_one({'_id': ObjectId(resume_id)})
    if not resume:
        return jsonify({'error': 'Candidate not found'}), 404

    cached = candidate_insights_collection.find_one({
        'job_id': job_id, 'resume_id': resume_id
    })
    if cached and cached.get('interview_questions'):
        return jsonify({'status': 'completed', 'result': cached['interview_questions']})

    from models.resume_model import Resume as ResumeModel
    from models.job_description_model import JobDescription

    def _generate():
        r = ResumeModel(**resume['parsed_data'])
        j = JobDescription(**job['parsed_data'])
        insight_doc = candidate_insights_collection.find_one({
            'job_id': job_id, 'resume_id': resume_id
        })
        return insight_service.generate_interview_questions(
            job_id, resume_id,
            insight_doc['skills_diff'], insight_doc['experience_diff'],
            insight_doc['education_diff'], r, j
        )

    bg_task_id = task_runner.submit(_generate)
    return jsonify({'status': 'running', 'task_id': bg_task_id})


@ai_engine_bp.route('/job/<job_id>/candidate/<resume_id>/chat', methods=['POST'])
@login_required
def insight_chat(job_id, resume_id):
    """Send a chat message about this candidate."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    data = request.get_json()
    if not data or not data.get('message'):
        return jsonify({'error': 'Message is required'}), 400

    insight = candidate_insights_collection.find_one({
        'job_id': job_id, 'resume_id': resume_id
    })
    if not insight:
        return jsonify({'error': 'Insight not found. Open the comparison page first.'}), 404

    try:
        result = chat_service.send_message(
            job_id, resume_id, current_user.id,
            data['message'], insight
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Failed to get response'}), 500


@ai_engine_bp.route('/job/<job_id>/candidate/<resume_id>/chat/history')
@login_required
def chat_history(job_id, resume_id):
    """Get chat history for this candidate insight session."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    messages = chat_service.get_chat_history(job_id, resume_id, current_user.id)
    return jsonify({'messages': messages})


@ai_engine_bp.route('/task/<task_id>/status')
@login_required
def check_task_status(task_id):
    """Poll background task status — checks MongoDB first, then in-memory TaskRunner."""
    if not isinstance(current_user, Recruiter):
        return jsonify({'error': 'Access denied'}), 403

    # Check MongoDB background_tasks collection (for CF-dispatched matching)
    bg_task = background_tasks_collection.find_one({'task_id': task_id})
    if bg_task:
        response = {'status': bg_task['status']}
        if bg_task['status'] == 'failed':
            response['error'] = bg_task.get('error', 'Unknown error')
        return jsonify(response)

    # Fall back to in-memory TaskRunner (for insight/question generation)
    status = task_runner.get_status(task_id)
    if not status:
        return jsonify({'error': 'Task not found'}), 404

    response = {'status': status['status']}
    if status['status'] == 'completed':
        response['result'] = status['result']
    elif status['status'] == 'failed':
        response['error'] = status['error']

    return jsonify(response)
