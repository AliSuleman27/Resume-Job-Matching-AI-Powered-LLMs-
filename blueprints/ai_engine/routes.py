import datetime
from datetime import timezone
import logging
from flask import render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user
from bson.objectid import ObjectId
from blueprints.ai_engine import ai_engine_bp
from blueprints.auth.user_models import Recruiter
from extensions import (
    jobs_collection, ai_results_collection, resumes_collection,
    applications_collection, matcher, task_runner
)
from services.type_convertor import prepare_document_for_mongodb

logger = logging.getLogger(__name__)


def _run_matching(job_id, recruiter_id):
    """Background task: run AI matching and store results in MongoDB."""
    ranked_candidates = matcher.match_all_applicants_for_job(job_id)

    total_candidates = len(ranked_candidates)
    high_score_candidates = len([c for c in ranked_candidates if c['overall_score'] >= 0.7])
    medium_score_candidates = len([c for c in ranked_candidates if 0.5 <= c['overall_score'] < 0.7])
    low_score_candidates = len([c for c in ranked_candidates if c['overall_score'] < 0.5])

    all_skills = []
    for candidate in ranked_candidates:
        if 'section_scores' in candidate and 'skills' in candidate['section_scores']:
            all_skills.extend(candidate.get('skills', []))

    skill_frequency = {}
    for skill in all_skills:
        skill_frequency[skill] = skill_frequency.get(skill, 0) + 1

    top_skills = sorted(skill_frequency.items(), key=lambda x: x[1], reverse=True)[:10]

    ai_results_data = {
        'job_id': ObjectId(job_id),
        'recruiter_id': ObjectId(recruiter_id),
        'ranked_candidates': ranked_candidates,
        'statistics': {
            'total_candidates': total_candidates,
            'high_score': high_score_candidates,
            'medium_score': medium_score_candidates,
            'low_score': low_score_candidates,
            'top_skills': top_skills
        },
        'created_at': datetime.datetime.now(timezone.utc),
        'updated_at': datetime.datetime.now(timezone.utc)
    }

    ai_results_data = prepare_document_for_mongodb(ai_results_data)

    existing_results = ai_results_collection.find_one({'job_id': ObjectId(job_id)})
    if existing_results:
        ai_results_collection.update_one(
            {'job_id': ObjectId(job_id)},
            {'$set': ai_results_data}
        )
    else:
        ai_results_collection.insert_one(ai_results_data)

    return {'total_candidates': total_candidates}


@ai_engine_bp.route('/job/<job_id>/run_ai_engine', methods=['POST'])
@login_required
def run_ai_engine(job_id):
    """Run AI inference for matching candidates to a specific job (background)"""
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

    try:
        existing_results = ai_results_collection.find_one({
            'job_id': ObjectId(job_id)
        })

        if existing_results:
            rerun_confirmed = request.form.get('confirm_rerun', 'false') == 'true'
            if not rerun_confirmed:
                flash('AI results already exist for this job. Please confirm if you want to rerun the analysis.', 'warning')
                return redirect(url_for('recruiter.recruiter_dashboard') + f'?show_rerun_modal={job_id}')

        # Submit to background task runner instead of blocking
        bg_task_id = task_runner.submit(_run_matching, job_id, current_user.id)
        logger.info(f"AI engine task {bg_task_id} started for job {job_id}")

        flash('AI analysis started. This may take a few minutes. Refresh the page to check for results.', 'success')
        return redirect(url_for('ai_engine.view_ai_results', job_id=job_id))

    except Exception as e:
        logger.error(f"Error starting AI engine for job {job_id}: {e}")
        flash('Error starting AI analysis. Please try again.', 'error')
        return redirect(url_for('recruiter.recruiter_dashboard'))


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
