import numpy as np
import os
import pickle
import faiss
import traceback
import logging
from typing import Dict, List, Any
from enum import Enum
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from models.job_description_model import JobDescription
from models.resume_model import Resume
from services.constraint_matcher import ConstraintMatcher

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
    skills: float = 0.25
    education: float = 0.15
    experience_title: float = 0.20
    experience_responsibilities: float = 0.20
    experience_skills: float = 0.10
    projects: float = 0.05
    summary: float = 0.05


class SemanticMatcher:
    def __init__(self, faiss_index_path: str = "./faiss_indexes", model=None):
        self.faiss_index_path = faiss_index_path
        self.model = model or SentenceTransformer("all-MiniLM-L6-v2")
        self.section_weights = SectionWeights()
        os.makedirs(faiss_index_path, exist_ok=True)
        self.indexes = {}
        self.id_mappings = {}
        self.processed_documents = set()
        self._load_or_create_indexes()

    def _load_or_create_indexes(self):
        for embedding_type in EmbeddingType:
            index_file = os.path.join(self.faiss_index_path, f"{embedding_type.value}.index")
            mapping_file = os.path.join(self.faiss_index_path, f"{embedding_type.value}_mapping.pkl")
            processed_file = os.path.join(self.faiss_index_path, f"{embedding_type.value}_processed.pkl")

            if os.path.exists(index_file) and os.path.exists(mapping_file):
                self.indexes[embedding_type.value] = faiss.read_index(index_file)
                with open(mapping_file, 'rb') as f:
                    self.id_mappings[embedding_type.value] = pickle.load(f)

                if os.path.exists(processed_file):
                    with open(processed_file, 'rb') as f:
                        processed_docs = pickle.load(f)
                        self.processed_documents.update(processed_docs)

                logger.info(f"Loaded existing index for {embedding_type.value}")
            else:
                self.indexes[embedding_type.value] = None
                self.id_mappings[embedding_type.value] = {}
                logger.info(f"Will create new index for {embedding_type.value}")

    def _is_document_processed(self, doc_id: str, doc_type: str) -> bool:
        return f"{doc_type}_{doc_id}" in self.processed_documents

    def _mark_document_processed(self, doc_id: str, doc_type: str):
        self.processed_documents.add(f"{doc_type}_{doc_id}")

    def _get_embedding(self, text: str, instruction: str = "") -> np.ndarray:
        if not text or text.strip() == "":
            return np.zeros(self.model.get_sentence_embedding_dimension())

        try:
            embedding = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.model.get_sentence_embedding_dimension())

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

        sections[EmbeddingType.EXPERIENCE_SKILLS.value] = sections[EmbeddingType.SKILLS.value]

        proj_texts = []
        if job.requirements:
            proj_texts.extend(job.requirements)
        if job.nice_to_have:
            proj_texts.extend(job.nice_to_have)
        sections[EmbeddingType.PROJECTS.value] = ". ".join(proj_texts)

        sections[EmbeddingType.SUMMARY.value] = job.summary or job.description or ""

        return sections

    def _add_to_faiss_index(self, embedding_type: str, embedding: np.ndarray, doc_id: str, doc_type: str):
        if self.indexes[embedding_type] is None:
            dimension = embedding.shape[0]
            self.indexes[embedding_type] = faiss.IndexFlatIP(dimension)

        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.reshape(1, -1).astype('float32')

        current_size = self.indexes[embedding_type].ntotal
        self.indexes[embedding_type].add(embedding)

        self.id_mappings[embedding_type][current_size] = {
            'id': doc_id,
            'type': doc_type
        }

    def _save_indexes(self):
        for embedding_type in EmbeddingType:
            if self.indexes[embedding_type.value] is not None:
                index_file = os.path.join(self.faiss_index_path, f"{embedding_type.value}.index")
                mapping_file = os.path.join(self.faiss_index_path, f"{embedding_type.value}_mapping.pkl")
                processed_file = os.path.join(self.faiss_index_path, f"{embedding_type.value}_processed.pkl")

                faiss.write_index(self.indexes[embedding_type.value], index_file)
                with open(mapping_file, 'wb') as f:
                    pickle.dump(self.id_mappings[embedding_type.value], f)
                with open(processed_file, 'wb') as f:
                    pickle.dump(list(self.processed_documents), f)

    def create_resume_embeddings(self, resume, resume_id: str):
        if self._is_document_processed(resume_id, 'resume'):
            logger.info(f"Resume {resume_id} already processed, skipping embedding generation")
            return

        logger.info(f"Creating embeddings for resume {resume_id}")

        sections = self._extract_resume_sections(resume)

        instructions = {
            EmbeddingType.SKILLS.value: "Represent the professional skills and competencies",
            EmbeddingType.EDUCATION.value: "Represent the educational background and qualifications",
            EmbeddingType.EXPERIENCE_TITLE.value: "Represent the job titles and career progression",
            EmbeddingType.EXPERIENCE_RESPONSIBILITIES.value: "Represent the work responsibilities and achievements",
            EmbeddingType.EXPERIENCE_SKILLS.value: "Represent the technical skills used in work experience",
            EmbeddingType.PROJECTS.value: "Represent the project experience and technical implementations",
            EmbeddingType.SUMMARY.value: "Represent the professional summary and career overview"
        }

        for section_type, text in sections.items():
            if text and text.strip():
                instruction = instructions[section_type]
                embedding = self._get_embedding(text, instruction)
                self._add_to_faiss_index(section_type, embedding, resume_id, 'resume')

        self._mark_document_processed(resume_id, 'resume')
        self._save_indexes()
        logger.info(f"Successfully created embeddings for resume {resume_id}")

    def create_job_embeddings(self, job, job_id: str):
        if self._is_document_processed(job_id, 'job'):
            logger.info(f"Job {job_id} already processed, skipping embedding generation")
            return

        logger.info(f"Creating embeddings for job {job_id}")

        sections = self._extract_job_sections(job)

        instructions = {
            EmbeddingType.SKILLS.value: "Represent the required skills and competencies for the job",
            EmbeddingType.EDUCATION.value: "Represent the educational requirements for the job",
            EmbeddingType.EXPERIENCE_TITLE.value: "Represent the job title and role requirements",
            EmbeddingType.EXPERIENCE_RESPONSIBILITIES.value: "Represent the job responsibilities and expectations",
            EmbeddingType.EXPERIENCE_SKILLS.value: "Represent the technical skills required for the job",
            EmbeddingType.PROJECTS.value: "Represent the project requirements and technical expectations",
            EmbeddingType.SUMMARY.value: "Represent the job summary and role overview"
        }

        for section_type, text in sections.items():
            if text and text.strip():
                instruction = instructions[section_type]
                embedding = self._get_embedding(text, instruction)
                self._add_to_faiss_index(section_type, embedding, job_id, 'job')

        self._mark_document_processed(job_id, 'job')
        self._save_indexes()
        logger.info(f"Successfully created embeddings for job {job_id}")

    def match_resume_and_job(self, resume, resume_id: str, job, job_id: str) -> Dict[str, Any]:
        logger.info(f"Matching resume {resume_id} with job {job_id}")

        resume_sections = self._extract_resume_sections(resume)
        job_sections = self._extract_job_sections(job)

        section_scores = {}

        instructions = {
            EmbeddingType.SKILLS.value: "Represent the professional skills and competencies",
            EmbeddingType.EDUCATION.value: "Represent the educational background and qualifications",
            EmbeddingType.EXPERIENCE_TITLE.value: "Represent the job titles and career progression",
            EmbeddingType.EXPERIENCE_RESPONSIBILITIES.value: "Represent the work responsibilities and achievements",
            EmbeddingType.EXPERIENCE_SKILLS.value: "Represent the technical skills used in work experience",
            EmbeddingType.PROJECTS.value: "Represent the project experience and technical implementations",
            EmbeddingType.SUMMARY.value: "Represent the professional summary and career overview"
        }

        for section_type in EmbeddingType:
            section_name = section_type.value
            resume_text = resume_sections.get(section_name, "")
            job_text = job_sections.get(section_name, "")

            if resume_text and job_text:
                instruction = instructions[section_name]
                resume_embedding = self._get_embedding(resume_text, instruction)
                job_embedding = self._get_embedding(job_text, instruction)

                similarity = cosine_similarity(
                    resume_embedding.reshape(1, -1),
                    job_embedding.reshape(1, -1)
                )[0][0]

                section_scores[section_name] = max(0.0, similarity)
            else:
                section_scores[section_name] = 0.0

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
    def __init__(self, constraint_matcher, mongodb_manager, model=None, faiss_index_path: str = "./faiss_indexes"):
        self.mongodb_manager = mongodb_manager
        self.semantic_matcher = SemanticMatcher(faiss_index_path=faiss_index_path, model=model)
        self.cm = constraint_matcher

    def process_resume(self, resume_id=None, jsonResume=None):
        try:
            pydantic_resume = None
            if jsonResume:
                pydantic_resume = Resume(**jsonResume)
            else:
                resume_data = self.mongodb_manager.get_resume_by_id(resume_id)['parsed_data']
                if not resume_data:
                    raise ValueError(f"Resume {resume_id} not found")
                pydantic_resume = Resume(**resume_data)
            self.semantic_matcher.create_resume_embeddings(pydantic_resume, resume_id)
            return True
        except Exception as e:
            logger.error(f"Error processing resume {resume_id}: {e}")
            return False

    def process_job(self, job_id=None, jsonJob=None):
        try:
            pydantic_job = None
            if jsonJob:
                pydantic_job = JobDescription(**jsonJob)
            else:
                job_data = self.mongodb_manager.get_job_by_id(job_id)
                if not job_data:
                    raise ValueError(f"Job {job_id} not found")
                job_data = job_data['parsed_data']
                pydantic_job = JobDescription(**job_data)

            self.semantic_matcher.create_job_embeddings(pydantic_job, job_id)
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

    def match_all_applicants_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Starting matching process for job {job_id}")

            job_data = self.mongodb_manager.get_job_by_id(job_id)
            if not job_data:
                raise ValueError(f"Job {job_id} not found")

            job_parsed_data = job_data['parsed_data']
            pydantic_job = JobDescription(**job_parsed_data)
            self.semantic_matcher.create_job_embeddings(pydantic_job, job_id)

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

                    self.semantic_matcher.create_resume_embeddings(pydantic_resume, resume_id)

                    applicant_name = pydantic_resume.basic_info.full_name

                    match_result = self.semantic_matcher.match_resume_and_job(
                        pydantic_resume, resume_id, pydantic_job, job_id
                    )
                    c_results = self.cm.calculate_overall_score(resume=pydantic_resume, job_description=pydantic_job)

                    applicant_result = {
                        'applicant_name': applicant_name,
                        'user_id': user_id,
                        'resume_id': resume_id,
                        'application_id': str(application['_id']),
                        'applied_at': application.get('applied_at'),
                        'status': application.get('status', 'submitted'),
                        'overall_score': 0.6 * match_result['overall_score'] + 0.4 * (c_results['overall']['score'] / 100),
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
                        'weights_used': match_result['weights_used']
                    }

                    results.append(applicant_result)
                    logger.info(f"Processed applicant {applicant_name} with score {applicant_result['overall_score']:.3f}")

                except Exception as e:
                    logger.error(f"Error processing application {application.get('_id')}: {e}")
                    logger.error(traceback.format_exc())
                    continue

            results.sort(key=lambda x: x['overall_score'], reverse=True)

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
