"""Google Cloud Function: AI Matching Pipeline.

HTTP-triggered, 2nd gen. Receives {task_id, job_id, recruiter_id},
runs the full matching pipeline, writes results to MongoDB.
"""

import os
import datetime
import logging
from datetime import timezone

import functions_framework
from pymongo import MongoClient
from bson.objectid import ObjectId

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Module-level globals (reused across warm invocations) ────────────
_mongo_client = None
_db = None
_matcher = None


def _get_db():
    global _mongo_client, _db
    if _db is None:
        _mongo_client = MongoClient(
            os.environ['MONGO_URI'],
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
        )
        _db = _mongo_client['resume_parser']
        # Init embedding cache for the embedding service
        from services.embedding_service import init_cache
        init_cache(_db['embedding_cache'])
    return _db


def _get_matcher():
    global _matcher
    if _matcher is None:
        db = _get_db()
        from services.hybrid_matcher import HybridMatcher
        from services.mongo_service import MongoDBManager
        from services.constraint_matcher import ConstraintMatcher
        cm = ConstraintMatcher()
        manager = MongoDBManager(db)
        _matcher = HybridMatcher(constraint_matcher=cm, mongodb_manager=manager)
    return _matcher


def _validate_auth(request):
    expected = os.environ.get('CF_AUTH_TOKEN', '')
    if not expected:
        return False
    auth_header = request.headers.get('Authorization', '')
    return auth_header == f'Bearer {expected}'


def _update_task(db, task_id, status, error=None):
    update = {
        '$set': {
            'status': status,
            'updated_at': datetime.datetime.now(timezone.utc),
        }
    }
    if error:
        update['$set']['error'] = str(error)
    db['background_tasks'].update_one({'task_id': task_id}, update)


def _run_matching(db, matcher, job_id, recruiter_id):
    """Run the full matching pipeline and store results."""
    from services.type_convertor import prepare_document_for_mongodb

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
            'top_skills': top_skills,
        },
        'created_at': datetime.datetime.now(timezone.utc),
        'updated_at': datetime.datetime.now(timezone.utc),
    }

    ai_results_data = prepare_document_for_mongodb(ai_results_data)

    ai_results_collection = db['ai_results_collection']
    existing = ai_results_collection.find_one({'job_id': ObjectId(job_id)})
    if existing:
        ai_results_collection.update_one(
            {'job_id': ObjectId(job_id)},
            {'$set': ai_results_data},
        )
    else:
        ai_results_collection.insert_one(ai_results_data)

    return {'total_candidates': total_candidates}


@functions_framework.http
def match_candidates(request):
    """GCF entry point — HTTP POST with JSON body."""
    if request.method == 'OPTIONS':
        return ('', 204, {'Access-Control-Allow-Origin': '*',
                          'Access-Control-Allow-Methods': 'POST',
                          'Access-Control-Allow-Headers': 'Authorization, Content-Type'})

    if request.method != 'POST':
        return ({'error': 'Method not allowed'}, 405)

    if not _validate_auth(request):
        return ({'error': 'Unauthorized'}, 401)

    data = request.get_json(silent=True)
    if not data:
        return ({'error': 'Missing JSON body'}, 400)

    task_id = data.get('task_id')
    job_id = data.get('job_id')
    recruiter_id = data.get('recruiter_id')

    if not all([task_id, job_id, recruiter_id]):
        return ({'error': 'task_id, job_id, and recruiter_id are required'}, 400)

    db = _get_db()
    matcher = _get_matcher()

    try:
        _run_matching(db, matcher, job_id, recruiter_id)
        _update_task(db, task_id, 'completed')
        logger.info(f"Matching completed for job {job_id}, task {task_id}")
        return ({'status': 'completed', 'task_id': task_id}, 200)

    except Exception as e:
        logger.error(f"Matching failed for job {job_id}, task {task_id}: {e}")
        _update_task(db, task_id, 'failed', error=e)
        return ({'status': 'failed', 'error': str(e)}, 500)
