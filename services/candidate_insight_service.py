import os
import json
import logging
from datetime import datetime, timezone

from groq import Groq
from models.resume_model import Resume
from models.job_description_model import JobDescription
from services.embedding_service import get_embedding, cosine_similarity
from services.constraint_matcher import ExperienceMatcher, EducationMatcher

logger = logging.getLogger(__name__)


class CandidateInsightService:
    """Core analysis engine for candidate-vs-JD deep comparison."""

    def __init__(self, insights_collection):
        self.insights_collection = insights_collection
        self.experience_matcher = ExperienceMatcher()
        self.education_matcher = EducationMatcher()

    # ------------------------------------------------------------------ #
    #  DIFF METHODS (zero LLM tokens)                                     #
    # ------------------------------------------------------------------ #

    def compute_skills_diff(self, resume: Resume, job: JobDescription) -> dict:
        """Compare JD skills vs candidate skills using fuzzy embedding match."""
        mandatory = list(job.skills.mandatory) if job.skills and job.skills.mandatory else []
        optional = list(job.skills.optional) if job.skills and job.skills.optional else []
        tools = list(job.skills.tools) if job.skills and job.skills.tools else []
        jd_skills = mandatory + optional + tools

        candidate_skills = [s.skill_name for s in (resume.skills or [])]

        # Also pull skills_used from experience
        for exp in (resume.experience or []):
            for s in (exp.skills_used or []):
                if s not in candidate_skills:
                    candidate_skills.append(s)

        # Pre-compute embeddings
        jd_embeddings = {s: get_embedding(s.lower()) for s in jd_skills} if jd_skills else {}
        cand_embeddings = {s: get_embedding(s.lower()) for s in candidate_skills} if candidate_skills else {}

        MATCH_THRESHOLD = 0.70

        matched = []
        missing_mandatory = []
        missing_optional = []
        jd_matched_set = set()
        cand_matched_set = set()

        for jd_skill in jd_skills:
            best_score = 0.0
            best_cand = None
            jd_vec = jd_embeddings[jd_skill]
            for cand_skill in candidate_skills:
                if cand_skill in cand_matched_set:
                    continue
                score = cosine_similarity(jd_vec, cand_embeddings[cand_skill])
                if score > best_score:
                    best_score = score
                    best_cand = cand_skill

            if best_score >= MATCH_THRESHOLD and best_cand:
                matched.append({
                    'jd_skill': jd_skill,
                    'candidate_skill': best_cand,
                    'similarity': round(best_score, 3)
                })
                jd_matched_set.add(jd_skill)
                cand_matched_set.add(best_cand)
            else:
                is_mandatory = jd_skill in mandatory
                if is_mandatory:
                    missing_mandatory.append(jd_skill)
                else:
                    missing_optional.append(jd_skill)

        extra = [s for s in candidate_skills if s not in cand_matched_set]

        return {
            'matched': matched,
            'missing_mandatory': missing_mandatory,
            'missing_optional': missing_optional,
            'extra': extra,
            'match_rate': round(len(matched) / len(jd_skills), 3) if jd_skills else 1.0
        }

    def compute_experience_diff(self, resume: Resume, job: JobDescription) -> dict:
        """Per-role relevance scoring reusing ExperienceMatcher logic."""
        if not resume.experience:
            return {'roles': [], 'total_relevant_years': 0, 'total_years': 0}

        job_title = job.title or ''
        similarities = self.experience_matcher.calculate_job_title_similarity(
            job_title, resume.experience
        )

        roles = []
        total_relevant = 0.0
        total_years = 0.0

        for exp_idx, sim_score, years in similarities:
            exp = resume.experience[exp_idx]
            is_relevant = sim_score >= self.experience_matcher.relevance_threshold
            roles.append({
                'job_title': exp.job_title,
                'company': exp.company or 'N/A',
                'start_date': exp.start_date or 'N/A',
                'end_date': exp.end_date or 'Present',
                'years': round(years, 1),
                'relevance_score': round(sim_score, 3),
                'is_relevant': is_relevant,
                'responsibilities': list(exp.responsibilities or [])[:3],
                'skills_used': list(exp.skills_used or [])
            })
            total_years += years
            if is_relevant:
                total_relevant += years

        # Get required years
        req_min = 0.0
        req_max = None
        if job.qualifications and job.qualifications.experience_years:
            req_min = job.qualifications.experience_years.min or 0.0
            req_max = job.qualifications.experience_years.max

        roles.sort(key=lambda r: r['relevance_score'], reverse=True)

        return {
            'roles': roles,
            'total_relevant_years': round(total_relevant, 1),
            'total_years': round(total_years, 1),
            'required_min': req_min,
            'required_max': req_max
        }

    def compute_education_diff(self, resume: Resume, job: JobDescription) -> dict:
        """Required vs actual education comparison."""
        required = []
        if job.qualifications and job.qualifications.education:
            for edu_req in job.qualifications.education:
                required.append({
                    'degree': edu_req.degree or 'N/A',
                    'field': edu_req.field_of_study or 'Any',
                    'level': edu_req.level.value if edu_req.level else None
                })

        actual = []
        for edu in (resume.education or []):
            actual.append({
                'degree': edu.degree,
                'field': edu.field or 'N/A',
                'institution': edu.institution or 'N/A',
                'grade': edu.grade,
                'start_date': edu.start_date,
                'end_date': edu.end_date
            })

        # Use education matcher for scoring
        if job.qualifications and job.qualifications.education:
            match_result = self.education_matcher.match_education(
                resume.education or [], job.qualifications.education
            )
            score = match_result.normalized_score
            explanation = match_result.explanation
        else:
            score = 0.8
            explanation = 'No specific education requirements'

        return {
            'required': required,
            'actual': actual,
            'score': round(score, 3),
            'explanation': explanation
        }

    def compute_certification_diff(self, resume: Resume, job: JobDescription) -> dict:
        """Required vs actual certifications."""
        required = []
        if job.qualifications and job.qualifications.certifications:
            required = list(job.qualifications.certifications)

        actual = []
        for cert in (resume.certifications or []):
            actual.append({
                'title': cert.title,
                'issuer': cert.issuer or 'N/A',
                'issue_date': cert.issue_date
            })

        actual_titles = [c['title'].lower() for c in actual]

        matched = []
        missing = []
        for req in required:
            req_lower = req.lower()
            found = any(req_lower in t or t in req_lower for t in actual_titles)
            if found:
                matched.append(req)
            else:
                missing.append(req)

        return {
            'required': required,
            'actual': actual,
            'matched': matched,
            'missing': missing
        }

    # ------------------------------------------------------------------ #
    #  LLM METHODS (lazy, cached)                                         #
    # ------------------------------------------------------------------ #

    def _build_compact_context(self, skills_diff: dict, experience_diff: dict,
                               education_diff: dict, resume: Resume,
                               job: JobDescription) -> str:
        """Build a compact structured context for LLM calls (~150 tokens)."""
        matched_skills = [m['jd_skill'] for m in skills_diff['matched']]
        missing_skills = skills_diff['missing_mandatory'] + skills_diff['missing_optional']

        lines = [
            f"JOB: {job.title}",
            f"CANDIDATE: {resume.basic_info.full_name}",
            f"SKILLS MATCH: {len(matched_skills)}/{len(matched_skills) + len(missing_skills)}",
            f"MATCHED: {', '.join(matched_skills[:8])}",
            f"MISSING: {', '.join(missing_skills[:8])}",
            f"EXTRA: {', '.join(skills_diff['extra'][:5])}",
            f"EXP: {experience_diff['total_relevant_years']}yr relevant / {experience_diff['total_years']}yr total",
            f"EDU: {education_diff['explanation']}",
        ]

        # Add top relevant roles
        relevant_roles = [r for r in experience_diff['roles'] if r['is_relevant']][:3]
        for r in relevant_roles:
            lines.append(f"ROLE: {r['job_title']} @ {r['company']} ({r['years']}yr, {r['relevance_score']} sim)")

        return "\n".join(lines)

    def generate_insights(self, job_id: str, resume_id: str,
                          skills_diff: dict, experience_diff: dict,
                          education_diff: dict, resume: Resume,
                          job: JobDescription) -> dict:
        """Groq LLM call for AI-powered candidate insights."""
        context = self._build_compact_context(
            skills_diff, experience_diff, education_diff, resume, job
        )

        prompt = f"""Analyze this candidate-job match and return JSON:
{context}

Return ONLY valid JSON with these keys:
- strengths (list of 3-5 strings)
- weaknesses (list of 2-4 strings)
- red_flags (list of 0-3 strings, empty if none)
- career_trajectory (string, 1-2 sentences)
- growth_potential (string, 1 sentence)
- skill_gap_analysis (string, 2-3 sentences)
- cultural_fit_signals (list of 1-3 strings)
- hiring_recommendation (string: "strong_hire", "hire", "maybe", "no_hire")
- confidence (float 0-1)"""

        try:
            client = Groq(api_key=os.getenv('GROQ_API_KEY'))
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a senior technical recruiter. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if hasattr(response, "usage") else 0

            # Parse JSON from response
            json_str = content.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start != -1 and end > start:
                json_str = json_str[start:end]

            insights = json.loads(json_str)
            insights['tokens_used'] = tokens

            # Save to MongoDB
            self.insights_collection.update_one(
                {'job_id': job_id, 'resume_id': resume_id},
                {'$set': {
                    'llm_insights': insights,
                    'insights_generated_at': datetime.now(timezone.utc)
                }}
            )

            return insights

        except Exception as e:
            logger.error(f"Error generating insights for {resume_id}: {e}")
            raise

    def generate_interview_questions(self, job_id: str, resume_id: str,
                                     skills_diff: dict, experience_diff: dict,
                                     education_diff: dict, resume: Resume,
                                     job: JobDescription) -> dict:
        """Groq LLM call for interview question generation."""
        context = self._build_compact_context(
            skills_diff, experience_diff, education_diff, resume, job
        )

        prompt = f"""Generate targeted interview questions for this candidate-job match:
{context}

Return ONLY valid JSON with these keys:
- skill_gap (list of objects with "question" and "purpose" keys, 3-4 items)
- behavioral (list of objects with "question" and "purpose" keys, 2-3 items)
- technical (list of objects with "question" and "purpose" keys, 3-4 items)
- red_flag_probes (list of objects with "question" and "purpose" keys, 1-2 items)"""

        try:
            client = Groq(api_key=os.getenv('GROQ_API_KEY'))
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a senior technical interviewer. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if hasattr(response, "usage") else 0

            json_str = content.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start != -1 and end > start:
                json_str = json_str[start:end]

            questions = json.loads(json_str)
            questions['tokens_used'] = tokens

            self.insights_collection.update_one(
                {'job_id': job_id, 'resume_id': resume_id},
                {'$set': {
                    'interview_questions': questions,
                    'questions_generated_at': datetime.now(timezone.utc)
                }}
            )

            return questions

        except Exception as e:
            logger.error(f"Error generating questions for {resume_id}: {e}")
            raise

    # ------------------------------------------------------------------ #
    #  ORCHESTRATOR                                                       #
    # ------------------------------------------------------------------ #

    def get_or_create_insight(self, job_id: str, resume_id: str,
                              resume_data: dict, job_data: dict) -> dict:
        """Compute diffs (fast), cache in MongoDB. LLM parts left null for lazy loading."""
        # Check cache first
        cached = self.insights_collection.find_one({
            'job_id': job_id, 'resume_id': resume_id
        })
        if cached:
            cached['_id'] = str(cached['_id'])
            return cached

        # Parse pydantic models
        resume = Resume(**resume_data)
        job = JobDescription(**job_data)

        # Compute diffs (fast, no LLM)
        skills_diff = self.compute_skills_diff(resume, job)
        experience_diff = self.compute_experience_diff(resume, job)
        education_diff = self.compute_education_diff(resume, job)
        certification_diff = self.compute_certification_diff(resume, job)

        doc = {
            'job_id': job_id,
            'resume_id': resume_id,
            'candidate_name': resume.basic_info.full_name,
            'job_title': job.title,
            'skills_diff': skills_diff,
            'experience_diff': experience_diff,
            'education_diff': education_diff,
            'certification_diff': certification_diff,
            'llm_insights': None,
            'interview_questions': None,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }

        result = self.insights_collection.insert_one(doc)
        doc['_id'] = str(result.inserted_id)
        return doc
