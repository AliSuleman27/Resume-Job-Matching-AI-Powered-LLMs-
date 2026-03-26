import numpy as np
import traceback
import logging
from typing import Dict, List, Any
from enum import Enum
from dataclasses import dataclass
from models.job_description_model import JobDescription
from models.resume_model import Resume
from services.constraint_matcher import ConstraintMatcher
from services.embedding_service import get_embedding, cosine_similarity, EMBEDDING_DIM
from services.questionnaire_scorer import score_answers
from services.skill_graph import get_skill_graph

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingType(Enum):
    SKILLS = "skills"
    EDUCATION = "education"
    EXPERIENCE_TITLE = "experience_title"
    EXPERIENCE_RESPONSIBILITIES = "experience_responsibilities"
    EXPERIENCE_SKILLS = "experience_skills"
    PROJECTS = "projects"
    SUMMARY = "summary"


@dataclass
class SectionWeights:
    # experience_skills was double-counting the skills section — weight set to 0.0
    # redistributed to skills (0.30) and experience_responsibilities (0.25)
    skills: float = 0.30
    education: float = 0.15
    experience_title: float = 0.20
    experience_responsibilities: float = 0.25
    experience_skills: float = 0.00
    projects: float = 0.05
    summary: float = 0.05


class SemanticMatcher:
    def __init__(self):
        self.section_weights = SectionWeights()

    def _get_embedding(self, text: str) -> np.ndarray:
        if not text or text.strip() == "":
            return np.zeros(EMBEDDING_DIM)

        try:
            embedding = get_embedding(text)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(EMBEDDING_DIM)

    def _compute_bm25_score(self, resume_text: str, jd_text: str) -> float:
        """BM25 keyword overlap score normalized to 0-1 range.

        Normalized by self-score so that a perfect keyword match returns 1.0.
        Falls back to 0.0 when rank-bm25 is not installed.
        """
        if not HAS_BM25 or not resume_text or not jd_text:
            return 0.0

        jd_tokens = jd_text.lower().split()
        resume_tokens = resume_text.lower().split()

        if not jd_tokens:
            return 0.0

        try:
            bm25 = BM25Okapi([jd_tokens])
            query_score = bm25.get_scores(resume_tokens)[0]
            self_score = bm25.get_scores(jd_tokens)[0]  # max possible score
            if self_score == 0:
                return 0.0
            return float(min(1.0, query_score / self_score))
        except Exception as e:
            logger.debug(f"BM25 scoring error: {e}")
            return 0.0

    def _compute_skills_score(self, resume: Resume, job: JobDescription) -> float:
        """Graph-based skill matching with BM25 keyword blend.

        Uses SkillGraph for multi-tier matching (exact/synonym/implies/hierarchy)
        instead of raw embedding cosine similarity.  Still blended with BM25
        keyword overlap (40/60 split).
        """
        if not job.skills:
            return 0.0

        mandatory = list(job.skills.mandatory or [])
        optional = list(job.skills.optional or [])
        tools = list(job.skills.tools or [])
        all_jd_skills = mandatory + optional + tools

        if not all_jd_skills:
            return 0.0

        candidate_skills = [s.skill_name for s in (resume.skills or [])]
        for exp in (resume.experience or []):
            for sk in (exp.skills_used or []):
                if sk not in candidate_skills:
                    candidate_skills.append(sk)

        if not candidate_skills:
            return 0.0

        sg = get_skill_graph()
        result = sg.match_skills_batch(mandatory, optional, tools, candidate_skills)
        graph_score = result.composite_score

        # BM25 keyword overlap as secondary signal
        jd_text = " ".join(all_jd_skills)
        resume_text = " ".join(candidate_skills)
        bm25_score = self._compute_bm25_score(resume_text, jd_text)

        return float(0.40 * bm25_score + 0.60 * graph_score)

    def _extract_resume_sections(self, resume) -> Dict[str, str]:
        sections = {}

        skills_text = ", ".join([skill.skill_name for skill in (resume.skills or [])])
        sections[EmbeddingType.SKILLS.value] = skills_text

        education_texts = []
        for edu in (resume.education or []):
            edu_text = f"{edu.degree} in {edu.field or 'N/A'} from {edu.institution or 'N/A'}"
            if edu.courses:
                edu_text += f" with courses: {', '.join(edu.courses)}"
            education_texts.append(edu_text)
        sections[EmbeddingType.EDUCATION.value] = ". ".join(education_texts)

        experience_titles = [exp.job_title for exp in (resume.experience or [])]
        sections[EmbeddingType.EXPERIENCE_TITLE.value] = ", ".join(experience_titles)

        all_responsibilities = []
        for exp in (resume.experience or []):
            all_responsibilities.extend(exp.responsibilities or [])
        sections[EmbeddingType.EXPERIENCE_RESPONSIBILITIES.value] = ". ".join(all_responsibilities)

        all_exp_skills = []
        for exp in (resume.experience or []):
            all_exp_skills.extend(exp.skills_used or [])
        sections[EmbeddingType.EXPERIENCE_SKILLS.value] = ", ".join(all_exp_skills)

        project_texts = []
        for project in (resume.projects or []):
            proj_text = f"{project.title}: {project.description or 'N/A'}"
            if project.technologies:
                proj_text += f" using {', '.join(project.technologies)}"
            project_texts.append(proj_text)
        sections[EmbeddingType.PROJECTS.value] = ". ".join(project_texts)

        sections[EmbeddingType.SUMMARY.value] = resume.summary or ""

        return sections

    def _extract_job_sections(self, job) -> Dict[str, str]:
        sections = {}

        skills_texts = []
        if job.skills:
            if job.skills.mandatory:
                skills_texts.extend(job.skills.mandatory)
            if job.skills.optional:
                skills_texts.extend(job.skills.optional)
            if job.skills.tools:
                skills_texts.extend(job.skills.tools)
        if job.nice_to_have:
            skills_texts.extend(job.nice_to_have)
        sections[EmbeddingType.SKILLS.value] = ", ".join(skills_texts)

        education_texts = []
        if job.qualifications and job.qualifications.education:
            for edu_req in job.qualifications.education:
                edu_text = f"{edu_req.degree or 'N/A'} in {edu_req.field_of_study or 'N/A'}"
                education_texts.append(edu_text)
        sections[EmbeddingType.EDUCATION.value] = ". ".join(education_texts)

        sections[EmbeddingType.EXPERIENCE_TITLE.value] = job.title

        resp_texts = []
        if job.responsibilities:
            resp_texts.extend(job.responsibilities)
        if job.description:
            resp_texts.append(job.description)
        sections[EmbeddingType.EXPERIENCE_RESPONSIBILITIES.value] = ". ".join(resp_texts)

        # Fixed: previously this was a direct copy of the skills section (double-counting).
        # Now extracts skills context from requirements and nice_to_have to represent
        # "skills expected in the role" as distinct from the formal skills list.
        req_skill_texts = []
        if job.requirements:
            req_skill_texts.extend(job.requirements)
        if job.nice_to_have:
            req_skill_texts.extend(job.nice_to_have)
        sections[EmbeddingType.EXPERIENCE_SKILLS.value] = ". ".join(req_skill_texts)

        proj_texts = []
        if job.requirements:
            proj_texts.extend(job.requirements)
        if job.nice_to_have:
            proj_texts.extend(job.nice_to_have)
        sections[EmbeddingType.PROJECTS.value] = ". ".join(proj_texts)

        sections[EmbeddingType.SUMMARY.value] = job.summary or job.description or ""

        return sections

    def match_resume_and_job(self, resume, resume_id: str, job, job_id: str) -> Dict[str, Any]:
        logger.info(f"Matching resume {resume_id} with job {job_id}")

        resume_sections = self._extract_resume_sections(resume)
        job_sections = self._extract_job_sections(job)

        section_scores = {}

        for section_type in EmbeddingType:
            section_name = section_type.value

            # --- Skills: per-skill embedding + BM25 hybrid ---
            if section_name == EmbeddingType.SKILLS.value:
                section_scores[section_name] = self._compute_skills_score(resume, job)
                continue

            resume_text = resume_sections.get(section_name, "")
            job_text = job_sections.get(section_name, "")

            if not resume_text or not job_text:
                section_scores[section_name] = 0.0
                continue

            # --- Responsibilities: BM25 + embedding hybrid ---
            if section_name == EmbeddingType.EXPERIENCE_RESPONSIBILITIES.value:
                resume_emb = self._get_embedding(resume_text)
                job_emb = self._get_embedding(job_text)
                emb_score = max(0.0, cosine_similarity(resume_emb, job_emb))
                bm25_score = self._compute_bm25_score(resume_text, job_text)
                section_scores[section_name] = float(0.40 * bm25_score + 0.60 * emb_score)
                continue

            # --- All other sections: standard embedding similarity ---
            resume_embedding = self._get_embedding(resume_text)
            job_embedding = self._get_embedding(job_text)
            similarity = cosine_similarity(resume_embedding, job_embedding)
            section_scores[section_name] = max(0.0, similarity)

        overall_score = (
            section_scores.get(EmbeddingType.SKILLS.value, 0) * self.section_weights.skills +
            section_scores.get(EmbeddingType.EDUCATION.value, 0) * self.section_weights.education +
            section_scores.get(EmbeddingType.EXPERIENCE_TITLE.value, 0) * self.section_weights.experience_title +
            section_scores.get(EmbeddingType.EXPERIENCE_RESPONSIBILITIES.value, 0) * self.section_weights.experience_responsibilities +
            section_scores.get(EmbeddingType.EXPERIENCE_SKILLS.value, 0) * self.section_weights.experience_skills +
            section_scores.get(EmbeddingType.PROJECTS.value, 0) * self.section_weights.projects +
            section_scores.get(EmbeddingType.SUMMARY.value, 0) * self.section_weights.summary
        )

        result = {
            'overall_score': overall_score,
            'section_scores': section_scores,
            'resume_id': resume_id,
            'job_id': job_id,
            'weights_used': {
                'skills': self.section_weights.skills,
                'education': self.section_weights.education,
                'experience_title': self.section_weights.experience_title,
                'experience_responsibilities': self.section_weights.experience_responsibilities,
                'experience_skills': self.section_weights.experience_skills,
                'projects': self.section_weights.projects,
                'summary': self.section_weights.summary
            }
        }

        logger.info(f"Match completed. Overall score: {overall_score:.3f}")
        return result


class HybridMatcher:
    def __init__(self, constraint_matcher, mongodb_manager):
        self.mongodb_manager = mongodb_manager
        self.semantic_matcher = SemanticMatcher()
        self.cm = constraint_matcher

    def process_resume(self, resume_id=None, jsonResume=None):
        """Validate resume can be parsed. No pre-computation needed without FAISS."""
        try:
            if jsonResume:
                Resume(**jsonResume)
            else:
                resume_data = self.mongodb_manager.get_resume_by_id(resume_id)['parsed_data']
                if not resume_data:
                    raise ValueError(f"Resume {resume_id} not found")
                Resume(**resume_data)
            return True
        except Exception as e:
            logger.error(f"Error processing resume {resume_id}: {e}")
            return False

    def process_job(self, job_id=None, jsonJob=None):
        """Validate job can be parsed. No pre-computation needed without FAISS."""
        try:
            if jsonJob:
                JobDescription(**jsonJob)
            else:
                job_data = self.mongodb_manager.get_job_by_id(job_id)
                if not job_data:
                    raise ValueError(f"Job {job_id} not found")
                job_data = job_data['parsed_data']
                JobDescription(**job_data)
            return True
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            return False

    def match_resume_to_job(self, resume_id: str, job_id: str) -> Dict[str, Any]:
        try:
            resume_data = self.mongodb_manager.get_resume_by_id(resume_id)
            job_data = self.mongodb_manager.get_job_by_id(job_id)

            if not resume_data:
                raise ValueError(f"Resume {resume_id} not found")
            if not job_data:
                raise ValueError(f"Job {job_id} not found")

            resume_data = resume_data['parsed_data']
            job_data = job_data['parsed_data']

            pydantic_resume = Resume(**resume_data)
            pydantic_job = JobDescription(**job_data)

            return self.semantic_matcher.match_resume_and_job(
                pydantic_resume, resume_id, pydantic_job, job_id
            )

        except Exception as e:
            logger.error(f"Error matching resume {resume_id} to job {job_id}: {e}")
            raise

    def _calibrate_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert raw overall scores to pool-relative percentile ranks.

        Results must already be sorted descending by overall_score.
        Rank 1 (best) → 1.0, rank n (worst) → 1/n.
        The original score is preserved as raw_score for transparency.

        This makes the >= 0.7 threshold mean "top 30% of this applicant pool"
        rather than an absolute quality score, giving recruiters much better
        differentiation across candidates.
        """
        n = len(results)
        if n == 0:
            return results
        if n == 1:
            results[0]['raw_score'] = round(results[0]['overall_score'], 4)
            results[0]['overall_score'] = 1.0
            return results

        for i, r in enumerate(results):
            r['raw_score'] = round(r['overall_score'], 4)
            r['overall_score'] = round((n - i) / n, 3)

        return results

    def match_all_applicants_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Starting matching process for job {job_id}")

            job_data = self.mongodb_manager.get_job_by_id(job_id)
            if not job_data:
                raise ValueError(f"Job {job_id} not found")

            job_parsed_data = job_data['parsed_data']
            pydantic_job = JobDescription(**job_parsed_data)

            # Check if job has screening questions for scoring
            screening_questions = job_data.get('screening_questions', [])
            has_questions = bool(screening_questions)

            applications = self.mongodb_manager.get_job_applicants(job_id)
            logger.info(f"Found {len(applications)} applicants for job {job_id}")

            if not applications:
                logger.info(f"No applicants found for job {job_id}")
                return []

            results = []

            for application in applications:
                try:
                    resume_id = str(application['resume_id'])
                    user_id = str(application['user_id'])

                    resume_data = self.mongodb_manager.get_resume_by_id(resume_id)
                    if not resume_data:
                        logger.warning(f"Resume {resume_id} not found for application")
                        continue

                    resume_parsed_data = resume_data['parsed_data']
                    pydantic_resume = Resume(**resume_parsed_data)

                    applicant_name = pydantic_resume.basic_info.full_name

                    match_result = self.semantic_matcher.match_resume_and_job(
                        pydantic_resume, resume_id, pydantic_job, job_id
                    )
                    c_results = self.cm.calculate_overall_score(resume=pydantic_resume, job_description=pydantic_job)

                    semantic_score = match_result['overall_score']
                    constraint_score = c_results['overall']['score'] / 100

                    # Questionnaire scoring: 50/30/20 when questions exist, 60/40 otherwise
                    q_score = 0.0
                    screening_answers = application.get('screening_answers', {})
                    if has_questions and screening_answers:
                        q_score = score_answers(screening_questions, screening_answers)
                        overall = 0.50 * semantic_score + 0.30 * constraint_score + 0.20 * q_score
                    else:
                        overall = 0.60 * semantic_score + 0.40 * constraint_score

                    applicant_result = {
                        'applicant_name': applicant_name,
                        'user_id': user_id,
                        'resume_id': resume_id,
                        'application_id': str(application['_id']),
                        'applied_at': application.get('applied_at'),
                        'status': application.get('status', 'submitted'),
                        'overall_score': overall,
                        'raw_score': None,  # populated by _calibrate_scores
                        'section_scores': {
                            'skills': match_result['section_scores'].get('skills', 0.0),
                            'education': match_result['section_scores'].get('education', 0.0),
                            'experience_title': match_result['section_scores'].get('experience_title', 0.0),
                            'experience_responsibilities': match_result['section_scores'].get('experience_responsibilities', 0.0),
                            'experience_skills': match_result['section_scores'].get('experience_skills', 0.0),
                            'projects': match_result['section_scores'].get('projects', 0.0),
                            'summary': match_result['section_scores'].get('summary', 0.0)
                        },
                        'constraint_results': c_results,
                        'cover_message': application.get('cover_message', ''),
                        'weights_used': match_result['weights_used'],
                        'questionnaire_score': q_score if has_questions else None,
                        'screening_answers': screening_answers if has_questions else None,
                    }

                    results.append(applicant_result)
                    logger.info(f"Processed applicant {applicant_name} with score {applicant_result['overall_score']:.3f}")

                except Exception as e:
                    logger.error(f"Error processing application {application.get('_id')}: {e}")
                    logger.error(traceback.format_exc())
                    continue

            # Sort by raw score then calibrate to pool-relative percentile ranks
            results.sort(key=lambda x: x['overall_score'], reverse=True)
            results = self._calibrate_scores(results)

            logger.info(f"Completed matching for job {job_id}. Processed {len(results)} applicants")
            return results

        except Exception as e:
            logger.error(f"Error in match_all_applicants_for_job for job {job_id}: {e}")
            raise

    def get_top_candidates(self, job_id: str, top_n: int = 10) -> List[Dict[str, Any]]:
        results = self.match_all_applicants_for_job(job_id)
        return results[:top_n]

    def get_candidates_above_threshold(self, job_id: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        results = self.match_all_applicants_for_job(job_id)
        return [result for result in results if result['overall_score'] >= threshold]
