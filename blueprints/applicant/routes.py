import os
import json
import logging
import datetime
from datetime import timezone
from io import BytesIO
from flask import render_template, request, jsonify, redirect, url_for, flash, send_file, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
from blueprints.applicant import applicant_bp
from blueprints.auth.user_models import Recruiter
from extensions import resumes_collection, applications_collection, jobs_collection, matcher
from services.llm_service import call_llm, extract_text_from_file, allowed_file
from services.get_doc_resume import parse_resume_from_dict, create_pretty_resume_docx
from services.get_doc_jd import parse_job_description_from_file, create_pretty_jd_docx
from services.resume_post_processor import post_process_resume

logger = logging.getLogger(__name__)


@applicant_bp.route("/")
def landing():
    if current_user.is_authenticated:
        if isinstance(current_user, Recruiter):
            return redirect(url_for('recruiter.recruiter_dashboard'))
        return redirect(url_for('applicant.dashboard'))
    return render_template('landing.html')


@applicant_bp.route('/upload_cv')
@login_required
def upload_cv():
    return render_template('index.html')


@applicant_bp.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)

            file_type = filename.rsplit('.', 1)[-1].lower()
            resume_text = extract_text_from_file(temp_path, file_type)

            llm_response = call_llm(resume_text)

            if not llm_response.get('output'):
                return jsonify({"error": "Failed to parse resume. Please try again."}), 500

            parsed_resume = json.loads(llm_response['output'])
            parsed_resume = post_process_resume(resume_text, parsed_resume)

            resume_data = {
                'user_id': ObjectId(current_user.id),
                'original_filename': filename,
                'parsed_data': parsed_resume,
                'raw_text': resume_text,
                'created_at': datetime.datetime.now(timezone.utc)
            }
            matcher.process_resume(jsonResume=parsed_resume)
            resumes_collection.insert_one(resume_data)

            try:
                os.remove(temp_path)
            except OSError:
                pass

            return jsonify({
                "status": "success",
                "parsed_resume": parsed_resume,
                "tokens_used": llm_response['tokens'],
                "filename": filename
            })

        except json.JSONDecodeError:
            logger.error("LLM returned invalid JSON for resume parsing")
            return jsonify({"error": "Failed to parse resume output. Please try again."}), 500
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return jsonify({"error": "An error occurred while processing your file."}), 500
    else:
        return jsonify({"error": "File type not allowed. Use PDF, DOCX, or TXT."}), 400


@applicant_bp.route('/dashboard')
@login_required
def dashboard():
    user_resumes = list(resumes_collection.find({'user_id': ObjectId(current_user.id)}))

    applications = list(applications_collection.aggregate([
        {'$match': {'user_id': ObjectId(current_user.id)}},
        {'$lookup': {'from': 'jobs', 'localField': 'job_id', 'foreignField': '_id', 'as': 'job'}},
        {'$unwind': '$job'},
        {'$sort': {'applied_at': -1}},
        {'$limit': 3}
    ]))

    return render_template('dashboard.html',
                           resumes=user_resumes,
                           applications=applications)


@applicant_bp.route('/resume/<resume_id>')
@login_required
def view_resume(resume_id):
    resume = resumes_collection.find_one({
        '_id': ObjectId(resume_id),
        'user_id': ObjectId(current_user.id)
    })

    if not resume:
        flash('Resume not found', 'error')
        return redirect(url_for('applicant.dashboard'))

    return render_template('resume_view.html', resume=resume)


@applicant_bp.route('/download_resume/<resume_id>', methods=['GET'])
@login_required
def download_resume(resume_id):
    try:
        resume_data = resumes_collection.find_one({
            '_id': ObjectId(resume_id),
            'user_id': ObjectId(current_user.id)
        })

        if not resume_data:
            flash('Resume not found', 'error')
            return redirect(url_for('applicant.dashboard'))

        resume_model = parse_resume_from_dict(resume_data['parsed_data'])

        buffer = BytesIO()
        create_pretty_resume_docx(resume_model, buffer)
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'{resume_model.basic_info.full_name} - Emplify-io.docx',
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except Exception as e:
        logger.error(f"Error generating DOCX: {e}")
        return jsonify({"error": "Failed to generate document."}), 500


@applicant_bp.route('/download_jd/<jd_id>', methods=['GET'])
@login_required
def download_jd(jd_id):
    try:
        job_data = jobs_collection.find_one({
            '_id': ObjectId(jd_id),
            'recruiter_id': ObjectId(current_user.id)
        })

        if not job_data:
            flash('Job Description not found', 'error')
            return redirect(url_for('applicant.dashboard'))

        job_model = parse_job_description_from_file(job_data['parsed_data'])

        buffer = BytesIO()
        create_pretty_jd_docx(job_model, buffer)
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'{job_model.title} - Emplify-io.docx',
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except Exception as e:
        logger.error(f"Error generating DOCX: {e}")
        return jsonify({"error": "Failed to generate document."}), 500
