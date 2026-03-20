import datetime
from datetime import timezone
from flask import render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user
from bson.objectid import ObjectId
from blueprints.jobs import jobs_bp
from extensions import jobs_collection, applications_collection, resumes_collection, ai_results_collection


@jobs_bp.route('/jobs')
@login_required
def browse_jobs():
    page = request.args.get('page', 1, type=int)
    per_page = 10

    jobs = list(jobs_collection.find({}).skip((page - 1) * per_page).limit(per_page))
    total_jobs = jobs_collection.count_documents({})

    return render_template('jobs/browse.html',
                           jobs=jobs,
                           current_page=page,
                           total_pages=(total_jobs // per_page) + 1)


@jobs_bp.route('/jobs/<job_id>')
@login_required
def job_detail(job_id):
    job = jobs_collection.find_one({'_id': ObjectId(job_id)})
    if not job:
        flash('Job not found', 'error')
        return redirect(url_for('jobs.browse_jobs'))

    has_applied = applications_collection.find_one({
        'user_id': ObjectId(current_user.id),
        'job_id': ObjectId(job_id)
    })

    user_resumes = list(resumes_collection.find({'user_id': ObjectId(current_user.id)}))

    return render_template('jobs/detail.html',
                           job=job,
                           has_applied=bool(has_applied),
                           resumes=user_resumes)


@jobs_bp.route('/jobs/<job_id>/apply', methods=['POST'])
@login_required
def apply_for_job(job_id):
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    resume_id = data.get('resume_id')
    cover_message = data.get('cover_message', '')

    job = jobs_collection.find_one({'_id': ObjectId(job_id)})
    if not job:
        return jsonify({"error": "Job not found"}), 404

    resume = resumes_collection.find_one({
        '_id': ObjectId(resume_id),
        'user_id': ObjectId(current_user.id)
    })
    if not resume:
        return jsonify({"error": "Resume not found"}), 404

    existing_application = applications_collection.find_one({
        'user_id': ObjectId(current_user.id),
        'job_id': ObjectId(job_id)
    })
    if existing_application:
        return jsonify({"error": "You've already applied for this job"}), 400

    application_data = {
        'user_id': ObjectId(current_user.id),
        'job_id': ObjectId(job_id),
        'resume_id': ObjectId(resume_id),
        'cover_message': cover_message,
        'status': 'submitted',
        'applied_at': datetime.datetime.now(timezone.utc),
        'updated_at': datetime.datetime.now(timezone.utc)
    }
    applications_collection.insert_one(application_data)

    return jsonify({
        "status": "success",
        "message": "Application submitted successfully"
    }), 200


@jobs_bp.route('/applications')
@login_required
def my_applications():
    applications = list(applications_collection.aggregate([
        {'$match': {'user_id': ObjectId(current_user.id)}},
        {'$lookup': {'from': 'jobs', 'localField': 'job_id', 'foreignField': '_id', 'as': 'job_info'}},
        {'$unwind': '$job_info'},
        {'$lookup': {'from': 'parsed_resumes', 'localField': 'resume_id', 'foreignField': '_id', 'as': 'resume_info'}},
        {'$unwind': '$resume_info'},
        {'$sort': {'applied_at': -1}},
        {'$project': {
            'status': 1,
            'applied_at': 1,
            'cover_message': 1,
            'feedback': 1,
            'feedback_at': 1,
            'job_title': '$job_info.parsed_data.title',
            'company': '$job_info.company',
            'resume_name': '$resume_info.original_filename',
            'job_id': '$job_info._id'
        }}
    ]))

    return render_template('applications/list.html', applications=applications)


@jobs_bp.route('/applications/<application_id>/results')
@login_required
def application_results(application_id):
    """View AI match results for a specific application"""
    application = applications_collection.find_one({
        '_id': ObjectId(application_id),
        'user_id': ObjectId(current_user.id)
    })

    if not application:
        flash('Application not found', 'error')
        return redirect(url_for('jobs.my_applications'))

    job = jobs_collection.find_one({'_id': application['job_id']})

    # Find AI results for this job and this user's resume
    ai_result = None
    ai_results_doc = ai_results_collection.find_one({'job_id': application['job_id']})
    if ai_results_doc:
        for candidate in ai_results_doc.get('ranked_candidates', []):
            if str(candidate.get('resume_id')) == str(application.get('resume_id')):
                ai_result = candidate
                break

    return render_template('applications/results.html',
                           application=application,
                           job=job,
                           ai_result=ai_result)
