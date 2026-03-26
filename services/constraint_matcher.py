import math
from typing import List, Optional, Dict, Tuple
import re
from datetime import datetime
from dateutil import parser
from services.embedding_service import get_embedding, cosine_similarity as emb_cosine_similarity
from services.skill_graph import get_skill_graph


class MatchScore:
    """Class to hold matching scores and explanations"""
    def __init__(self, score: float = 0.0, max_score: float = 1.0, explanation: str = "", details: dict = None):
        self.score = score
        self.max_score = max_score
        self.normalized_score = score / max_score if max_score > 0 else 0.0
        self.explanation = explanation
        self.details = details or {}


class EducationMatcher:
    """Handles education-based matching between resume and job description"""
    # Degree hierarchy mapping (higher number = higher level)
    DEGREE_HIERARCHY = {
        'high_school': 1, 'diploma': 1, 'certificate': 1,
        'associate': 2, 'associates': 2,
        'bachelor': 3, 'bachelors': 3, 'bsc': 3, 'ba': 3, 'btech': 3, 'be': 3,
        'master': 4, 'masters': 4, 'msc': 4, 'ma': 4, 'mtech': 4, 'mba': 4,
        'phd': 5, 'doctorate': 5, 'doctoral': 5
    }

    # Grade conversion to 4.0 scale (approximate)
    GRADE_CONVERSIONS = {
        'a+': 4.0, 'a': 3.7, 'a-': 3.3,
        'b+': 3.0, 'b': 2.7, 'b-': 2.3,
        'c+': 2.0, 'c': 1.7, 'c-': 1.3,
        'd': 1.0, 'f': 0.0
    }

    # Field similarity groups (can be expanded with embeddings)
    FIELD_SIMILARITY = {
        'computer_science': ['computer science', 'cs', 'software engineering', 'information technology', 'it',
                             'ai', 'artificial intelligence', 'intelligence', 'data science', 'machine learning',
                             'cybersecurity', 'information systems', 'computing'],
        'engineering': ['engineering', 'mechanical', 'electrical', 'civil', 'chemical', 'systems engineering'],
        'business': ['business', 'mba', 'management', 'finance', 'marketing', 'economics', 'accounting'],
        'science': ['physics', 'chemistry', 'biology', 'mathematics', 'statistics', 'math', 'maths'],
        'design': ['design', 'graphic design', 'ui/ux', 'visual design', 'product design', 'ux', 'ui']
    }

    def __init__(self):
        pass

    def embed_text(self, text: str):
        return get_embedding(text)

    def normalize_degree(self, degree: str) -> str:
        """Normalize degree name for comparison"""
        if not degree:
            return ""
        degree_clean = re.sub(r'[^\w\s]', '', degree.lower().strip())
        for key in self.DEGREE_HIERARCHY:
            if key in degree_clean:
                return key
        return degree_clean

    def normalize_field(self, field: str) -> str:
        """Normalize field of study for comparison"""
        if not field:
            return ""
        return re.sub(r'[^\w\s]', '', field.lower().strip())

    def get_degree_level(self, degree: str) -> int:
        """Get numeric level of degree"""
        normalized = self.normalize_degree(degree)
        return self.DEGREE_HIERARCHY.get(normalized, 0)

    def convert_grade_to_gpa(self, grade: str) -> Optional[float]:
        """Convert various grade formats to 4.0 GPA scale"""
        if not grade:
            return None

        grade_clean = grade.lower().strip()

        gpa_match = re.search(r'(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?', grade_clean)
        if gpa_match:
            gpa = float(gpa_match.group(1))
            scale = float(gpa_match.group(2)) if gpa_match.group(2) else 4.0
            return min(4.0, (gpa / scale) * 4.0)

        if grade_clean in self.GRADE_CONVERSIONS:
            return self.GRADE_CONVERSIONS[grade_clean]

        percentage_match = re.search(r'(\d+)%?', grade_clean)
        if percentage_match:
            percentage = float(percentage_match.group(1))
            if percentage >= 90:
                return 4.0
            elif percentage >= 80:
                return 3.0
            elif percentage >= 70:
                return 2.0
            elif percentage >= 60:
                return 1.0
            else:
                return 0.5

        return None

    def calculate_field_similarity(self, resume_field: str, required_field: str) -> float:
        """Calculate semantic similarity between fields of study"""
        if not resume_field or not required_field:
            return 0.0

        resume_field_norm = self.normalize_field(resume_field)
        required_field_norm = self.normalize_field(required_field)

        if resume_field_norm == required_field_norm:
            return 1.0

        group_score = 0.0
        for group, fields in self.FIELD_SIMILARITY.items():
            resume_in_group = any(field in resume_field_norm for field in fields)
            required_in_group = any(field in required_field_norm for field in fields)
            if resume_in_group and required_in_group:
                group_score = 0.8
                break

        resume_words = set(resume_field_norm.split())
        required_words = set(required_field_norm.split())
        intersection = resume_words & required_words
        union = resume_words | required_words
        overlap_score = (len(intersection) / len(union)) * 0.6 if union else 0.0

        try:
            resume_vec = self.embed_text(resume_field_norm)
            required_vec = self.embed_text(required_field_norm)
            emb_score = emb_cosine_similarity(resume_vec, required_vec)
        except Exception:
            emb_score = 0.0

        total_score = (group_score * 0.5) + (overlap_score * 0.2) + (emb_score * 0.4)
        return min(total_score, 0.99)

    def match_education(self, resume_education: List, job_education: List) -> MatchScore:
        """
        Match education requirements between resume and job description

        Scoring Rules:
        1. Degree Level Match (40 points)
        2. Field Relevance (35 points)
        3. Grade Quality (25 points)
        """
        if not job_education:
            return MatchScore(80.0, 100.0, "No specific education requirements")

        if not resume_education:
            return MatchScore(0.0, 100.0, "No education information in resume")

        best_match_score = 0.0
        best_explanation = ""

        for job_req in job_education:
            for resume_edu in resume_education:
                current_score = 0.0
                explanations = []

                # 1. Degree Level Matching (40 points)
                resume_level = self.get_degree_level(resume_edu.degree)
                if job_req.degree:
                    required_level = self.get_degree_level(job_req.degree)
                elif job_req.level:
                    required_level = self.DEGREE_HIERARCHY.get(job_req.level.value, 0)
                else:
                    required_level = 3  # Default to bachelor's

                if resume_level == required_level:
                    current_score += 40
                    explanations.append(f"Exact degree level match ({resume_edu.degree})")
                elif resume_level > required_level:
                    current_score += 35
                    explanations.append(f"Higher degree than required ({resume_edu.degree})")
                elif resume_level == required_level - 1:
                    current_score += 25
                    explanations.append(f"One level below required degree")
                elif resume_level > 0:
                    current_score += 10
                    explanations.append(f"Lower degree level")
                else:
                    explanations.append(f"No recognized degree level")

                # 2. Field Relevance (35 points)
                field_similarity = 0.0
                if job_req.field_of_study and resume_edu.field:
                    field_similarity = self.calculate_field_similarity(
                        resume_edu.field, job_req.field_of_study
                    )
                elif not job_req.field_of_study:
                    field_similarity = 0.8

                field_score = field_similarity * 35
                current_score += field_score

                if field_similarity >= 0.9:
                    explanations.append("Excellent field match")
                elif field_similarity >= 0.7:
                    explanations.append("Good field relevance")
                elif field_similarity >= 0.4:
                    explanations.append("Moderate field relevance")
                elif field_similarity > 0:
                    explanations.append("Some field relevance")
                else:
                    explanations.append("Field not closely related")

                # 3. Grade Quality (25 points)
                if resume_edu.grade:
                    gpa = self.convert_grade_to_gpa(resume_edu.grade)
                    if gpa:
                        if gpa >= 3.5:
                            current_score += 25
                            explanations.append(f"Excellent grades (GPA: {gpa:.1f})")
                        elif gpa >= 3.0:
                            current_score += 20
                            explanations.append(f"Good grades (GPA: {gpa:.1f})")
                        elif gpa >= 2.5:
                            current_score += 15
                            explanations.append(f"Average grades (GPA: {gpa:.1f})")
                        else:
                            current_score += 10
                            explanations.append(f"Below average grades (GPA: {gpa:.1f})")
                    else:
                        current_score += 12
                        explanations.append("Grade format not recognized")
                else:
                    current_score += 12
                    explanations.append("No grade information")

                if current_score > best_match_score:
                    best_match_score = current_score
                    best_explanation = "; ".join(explanations)

        return MatchScore(best_match_score, 100.0, best_explanation)


class LocationMatcher:
    """Handles location-based matching between resume and job description"""

    def __init__(self):
        pass

    def normalize_location(self, location_str: str) -> str:
        if not location_str:
            return ""
        return re.sub(r'[^\w\s]', '', location_str.lower().strip())

    def extract_resume_locations(self, resume) -> List[str]:
        locations = []

        if resume.contact_info.address:
            addr = resume.contact_info.address
            if addr.city:
                locations.append(addr.city)
            if addr.state:
                locations.append(addr.state)
            if addr.country:
                locations.append(addr.country)

        for exp in resume.experience:
            if exp.location:
                locations.append(exp.location)

        for edu in resume.education:
            if edu.location:
                locations.append(edu.location)

        return [self.normalize_location(loc) for loc in locations if loc]

    def check_location_match(self, resume_locations: List[str], job_location: str) -> bool:
        job_loc_norm = self.normalize_location(job_location)
        for resume_loc in resume_locations:
            if resume_loc in job_loc_norm or job_loc_norm in resume_loc:
                return True
        return False

    def match_location(self, resume, job_description) -> MatchScore:
        max_score = 100.0

        if job_description.is_remote:
            return MatchScore(max_score, max_score, "Remote work - location not relevant")

        resume_locations = self.extract_resume_locations(resume)

        job_locations = []
        if job_description.locations:
            for loc in job_description.locations:
                if loc.city:
                    job_locations.append(loc.city)
                if loc.state:
                    job_locations.append(loc.state)
                if loc.country:
                    job_locations.append(loc.country)

        job_locations = [self.normalize_location(loc) for loc in job_locations]

        if not job_locations:
            return MatchScore(max_score, max_score, "No specific location requirements")

        if not resume_locations:
            penalty_score = max_score - 30
            return MatchScore(penalty_score, max_score,
                              "No location information in resume - 30 point penalty")

        location_matches = []
        for job_loc in job_locations:
            for resume_loc in resume_locations:
                if self.check_location_match([resume_loc], job_loc):
                    location_matches.append((resume_loc, job_loc))

        if location_matches:
            match_details = ", ".join([f"{r} matches {j}" for r, j in location_matches[:2]])
            return MatchScore(max_score, max_score, f"Location match found: {match_details}")
        else:
            penalty_score = max_score - 20
            return MatchScore(penalty_score, max_score,
                              "No location match - 20 point penalty")


class ExperienceMatcher:
    """Handles experience-based matching between resume and job description"""

    def __init__(self):
        self.relevance_threshold = 0.5

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        try:
            vec1 = get_embedding(text1)
            vec2 = get_embedding(text2)
            similarity = emb_cosine_similarity(vec1, vec2)
            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.0

    def normalize_job_title(self, title: str) -> str:
        if not title:
            return ""
        normalized = re.sub(r'[^\w\s]', ' ', title.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in normalized.split() if word not in stop_words]
        return ' '.join(words).strip()

    def parse_date(self, date_str: str) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            if date_str.lower() in ['present', 'current', 'now', 'ongoing']:
                return datetime.now()
            return parser.parse(date_str, fuzzy=True)
        except Exception:
            return None

    def calculate_experience_duration(self, start_date: str, end_date: str) -> float:
        """Calculate raw experience duration in years (no decay)."""
        start = self.parse_date(start_date)
        end = self.parse_date(end_date)

        if not start:
            return 0.0
        if not end:
            end = datetime.now()

        duration = (end - start).days / 365.25
        return max(0.0, duration)

    def _calculate_weighted_duration(self, start_date: str, end_date: str) -> float:
        """Calculate experience duration weighted by recency using exponential decay.

        Recent roles contribute their full duration. Older roles are discounted:
          weight = exp(-0.15 * years_since_end)
        This gives ~50% weight to a role that ended 4.6 years ago and ~87%
        weight to one that ended 1 year ago, making recent experience matter more.
        """
        start = self.parse_date(start_date)
        end = self.parse_date(end_date)

        if not start:
            return 0.0

        now = datetime.now()
        if not end:
            end = now

        raw_duration = (end - start).days / 365.25
        if raw_duration <= 0:
            return 0.0

        years_since_end = max(0.0, (now - end).days / 365.25)
        decay = math.exp(-0.15 * years_since_end)

        return max(0.0, raw_duration * decay)

    def calculate_job_title_similarity(self, job_title: str, resume_experiences: List) -> List[Tuple[int, float, float]]:
        """
        Calculate similarity between job title and all resume experiences.
        Returns: List of (experience_index, similarity_score, recency_weighted_years)
        """
        job_title_norm = self.normalize_job_title(job_title)
        similarities = []

        for i, exp in enumerate(resume_experiences):
            exp_title_norm = self.normalize_job_title(exp.job_title)
            similarity = self.calculate_semantic_similarity(job_title_norm, exp_title_norm)
            # Use recency-weighted duration instead of raw duration
            years = self._calculate_weighted_duration(exp.start_date, exp.end_date)
            similarities.append((i, similarity, years))

        return similarities

    def aggregate_relevant_experience(self, similarities: List[Tuple[int, float, float]]) -> Tuple[float, List[Tuple[int, float]]]:
        """
        Aggregate only relevant experience based on relevance threshold.
        Returns: (relevant_weighted_years, relevant_jobs_details)
        """
        relevant_years = 0.0
        relevant_jobs = []

        for exp_idx, similarity, years in similarities:
            if similarity >= self.relevance_threshold:
                relevant_years += years
                relevant_jobs.append((exp_idx, similarity))

        return relevant_years, relevant_jobs

    def calculate_experience_score(self, relevant_years: float, required_min: float,
                                   required_max: Optional[float] = None) -> Tuple[float, str]:
        """Calculate experience score based only on relevant recency-weighted experience."""
        if required_max is None:
            required_max = required_min * 1.5

        max_score = 100.0

        if relevant_years >= required_min:
            if relevant_years >= required_max:
                final_score = max_score
            else:
                progress = (relevant_years - required_min) / (required_max - required_min)
                final_score = 60.0 + (progress * 40.0)
        else:
            if required_min > 0:
                final_score = (relevant_years / required_min) * 60.0
            else:
                final_score = 60.0

        final_score = min(max_score, final_score)

        explanation_parts = [
            f"Relevant experience: {relevant_years:.1f} yrs (recency-weighted)",
            f"Required: {required_min:.1f}-{required_max:.1f} yrs"
        ]

        if relevant_years >= required_min:
            explanation_parts.append("Meets minimum requirement")
        else:
            shortage = required_min - relevant_years
            explanation_parts.append(f"{shortage:.1f} yrs short of minimum")

        return final_score, "; ".join(explanation_parts)

    def match_experience(self, resume_experiences: List, job_description) -> MatchScore:
        if not resume_experiences:
            return MatchScore(0.0, 100.0, "No work experience found in resume")

        required_min = 0.0
        required_max = None

        if (job_description.qualifications and
                job_description.qualifications.experience_years):
            exp_req = job_description.qualifications.experience_years
            required_min = exp_req.min or 0.0
            required_max = exp_req.max

        if required_min == 0.0 and job_description.job_level:
            level_requirements = {
                'entry': (0, 2),
                'mid': (2, 3),
                'senior': (5, 8),
                'lead': (7, 10),
                'manager': (5, 12),
                'director': (8, 15),
                'executive': (10, 20)
            }
            if job_description.job_level.value in level_requirements:
                required_min, required_max = level_requirements[job_description.job_level.value]

        if required_min == 0.0:
            required_min = 1.0

        job_title = job_description.title
        similarities = self.calculate_job_title_similarity(job_title, resume_experiences)
        relevant_years, relevant_jobs = self.aggregate_relevant_experience(similarities)

        score, explanation = self.calculate_experience_score(
            relevant_years, required_min, required_max
        )

        if relevant_jobs:
            job_details = []
            for exp_idx, similarity in relevant_jobs[:3]:
                exp = resume_experiences[exp_idx]
                job_details.append(f"{exp.job_title} ({similarity:.2f} sim)")
            explanation += f"; Relevant roles: {', '.join(job_details)}"
        elif relevant_years == 0:
            explanation += "; No relevant experience found"

        return MatchScore(score, 100.0, explanation)


class SkillsMatcher:
    """Graph-based skill matching for hard constraint scoring.

    Uses SkillGraph for multi-tier matching (exact/synonym/implies/hierarchy)
    with embedding fallback capped at 0.55 for unknown skills.
    Mandatory skills are weighted 75%, optional/tools 25%.
    """

    def __init__(self):
        pass

    def match_skills(self, resume, job_description) -> MatchScore:
        if not job_description.skills:
            return MatchScore(80.0, 100.0, "No specific skill requirements listed")

        mandatory = list(job_description.skills.mandatory or [])
        optional = list(job_description.skills.optional or [])
        tools = list(job_description.skills.tools or [])
        all_jd_skills = mandatory + optional + tools

        if not all_jd_skills:
            return MatchScore(80.0, 100.0, "No specific skill requirements listed")

        candidate_skills = [s.skill_name for s in (resume.skills or [])]
        for exp in (resume.experience or []):
            for sk in (exp.skills_used or []):
                if sk not in candidate_skills:
                    candidate_skills.append(sk)

        if not candidate_skills:
            return MatchScore(0.0, 100.0, "No skills found in resume")

        sg = get_skill_graph()
        result = sg.match_skills_batch(mandatory, optional, tools, candidate_skills)

        matched_count = len(result.matched_details)
        mandatory_matched = sum(
            1 for d in result.matched_details if d.jd_skill in mandatory
        )
        optional_matched = matched_count - mandatory_matched

        final_score = result.composite_score * 100

        explanation = (
            f"Matched {matched_count}/{len(all_jd_skills)} skills "
            f"(mandatory: {mandatory_matched}/{len(mandatory)}, "
            f"optional: {optional_matched}/{len(optional) + len(tools)})"
        )
        if result.missing_mandatory:
            explanation += f"; Missing mandatory: {', '.join(result.missing_mandatory[:4])}"

        skill_details = {
            'matched': [
                {
                    'jd_skill': d.jd_skill,
                    'candidate_skill': d.candidate_skill,
                    'similarity': d.similarity,
                    'tier': d.tier,
                    'is_mandatory': d.jd_skill in mandatory,
                }
                for d in result.matched_details
            ],
            'missing_mandatory': result.missing_mandatory,
            'missing_optional': result.missing_optional,
            'extra_skills': result.extra_skills,
            'mandatory_rate': result.mandatory_rate,
            'optional_rate': result.optional_rate,
        }

        return MatchScore(final_score, 100.0, explanation, details=skill_details)


class ConstraintMatcher:
    """Main class that combines all matching algorithms"""

    def __init__(self):
        self.education_matcher = EducationMatcher()
        self.location_matcher = LocationMatcher()
        self.experience_matcher = ExperienceMatcher()
        self.skills_matcher = SkillsMatcher()

    def calculate_overall_score(self, resume, job_description, weights: Dict[str, float] = None) -> Dict:
        """
        Calculate overall matching score between resume and job description.

        Weights (updated — skills now fully implemented):
          skills:     40%  — per-skill embedding coverage (mandatory vs optional)
          experience: 35%  — recency-weighted relevant years vs required years
          education:  15%  — degree level + field relevance + grades
          location:   10%  — city/country match or remote bypass
        """
        if weights is None:
            weights = {
                'skills': 0.40,
                'experience': 0.35,
                'education': 0.15,
                'location': 0.10,
            }

        results = {}

        # Skills matching
        skills_score = self.skills_matcher.match_skills(resume, job_description)
        results['skills'] = {
            'score': skills_score.score,
            'max_score': skills_score.max_score,
            'normalized_score': skills_score.normalized_score,
            'explanation': skills_score.explanation,
            'weight': weights['skills'],
            'skill_match_details': skills_score.details,
        }

        # Education matching
        if job_description.qualifications and job_description.qualifications.education:
            education_score = self.education_matcher.match_education(
                resume.education,
                job_description.qualifications.education
            )
        else:
            education_score = MatchScore(80.0, 100.0, "No specific education requirements")

        results['education'] = {
            'score': education_score.score,
            'max_score': education_score.max_score,
            'normalized_score': education_score.normalized_score,
            'explanation': education_score.explanation,
            'weight': weights['education']
        }

        # Location matching
        location_score = self.location_matcher.match_location(resume, job_description)
        results['location'] = {
            'score': location_score.score,
            'max_score': location_score.max_score,
            'normalized_score': location_score.normalized_score,
            'explanation': location_score.explanation,
            'weight': weights['location']
        }

        # Experience matching
        experience_score = self.experience_matcher.match_experience(resume.experience, job_description)
        results['experience'] = {
            'score': experience_score.score,
            'max_score': experience_score.max_score,
            'normalized_score': experience_score.normalized_score,
            'explanation': experience_score.explanation,
            'weight': weights['experience']
        }

        # Weighted overall score
        total_weighted_score = sum(
            results[component]['normalized_score'] * results[component]['weight']
            for component in results
        )

        results['overall'] = {
            'score': total_weighted_score * 100,
            'max_score': 100.0,
            'normalized_score': total_weighted_score,
            'breakdown': {k: v for k, v in results.items() if k != 'overall'}
        }

        return results
