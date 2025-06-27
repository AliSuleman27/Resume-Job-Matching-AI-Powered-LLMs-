

from typing import List, Optional, Dict, Tuple
import re
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from dateutil import parser
from typing import List, Optional, Dict, Tuple

class MatchScore:
    """Class to hold matching scores and explanations"""
    def __init__(self, score: float = 0.0, max_score: float = 1.0, explanation: str = ""):
        self.score = score
        self.max_score = max_score
        self.normalized_score = score / max_score if max_score > 0 else 0.0
        self.explanation = explanation

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
        'computer_science': ['computer science', 'cs', 'software engineering', 'information technology', 'it', 'ai', 'artificial intelligence', 'intelligence'],
        'engineering': ['engineering', 'mechanical', 'electrical', 'civil', 'chemical'],
        'business': ['business', 'mba', 'management', 'finance', 'marketing', 'economics'],
        'science': ['physics', 'chemistry', 'biology', 'mathematics', 'statistics'],
        'design': ['design', 'graphic design', 'ui/ux', 'visual design', 'product design']
    }
    
    def __init__(self,model):
        self.model = model

    def embed_text(self, text: str):
        return self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    
    def normalize_degree(self, degree: str) -> str:
        """Normalize degree name for comparison"""
        if not degree:
            return ""
        degree_clean = re.sub(r'[^\w\s]', '', degree.lower().strip())
        # Extract key degree terms
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
        
        # Direct GPA (e.g., "3.5", "3.5/4.0")
        gpa_match = re.search(r'(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?', grade_clean)
        if gpa_match:
            gpa = float(gpa_match.group(1))
            scale = float(gpa_match.group(2)) if gpa_match.group(2) else 4.0
            return min(4.0, (gpa / scale) * 4.0)
        
        # Letter grades
        if grade_clean in self.GRADE_CONVERSIONS:
            return self.GRADE_CONVERSIONS[grade_clean]
        
        # Percentage (assuming 90+ = A, 80+ = B, etc.)
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

        # Exact match
        if resume_field_norm == required_field_norm:
            return 1.0

        # Group similarity
        group_score = 0.0
        for group, fields in self.FIELD_SIMILARITY.items():
            resume_in_group = any(field in resume_field_norm for field in fields)
            required_in_group = any(field in required_field_norm for field in fields)
            if resume_in_group and required_in_group:
                group_score = 0.8
                break

        # Partial word overlap (Jaccard)
        resume_words = set(resume_field_norm.split())
        required_words = set(required_field_norm.split())
        intersection = resume_words & required_words
        union = resume_words | required_words
        overlap_score = (len(intersection) / len(union)) * 0.6 if union else 0.0

        # Embedding similarity using SentenceTransformer
        try:
            resume_vec = self.embed_text(resume_field_norm)
            required_vec = self.embed_text(required_field_norm)
            emb_score = float(cosine_similarity([resume_vec], [required_vec])[0][0])
        except Exception:
            emb_score = 0.0

        # Combine scores: you can tune these weights
        total_score = (group_score * 0.5) + (overlap_score * 0.2) + (emb_score * 0.4)
        return min(total_score, 0.99)  # never return 1.0 unless exact match

    def match_education(self, resume_education: List, job_education: List) -> MatchScore:
        """
        Match education requirements between resume and job description
        
        Scoring Rules:
        1. Degree Level Match (40 points):
           - Exact match: 40 points
           - Higher degree: 35 points
           - One level lower: 25 points
           - Two+ levels lower: 10 points
           - No relevant degree: 0 points
        
        2. Field Relevance (35 points):
           - Exact field match: 35 points
           - Closely related: 28 points
           - Somewhat related: 15 points
           - Unrelated: 5 points
        
        3. Grade Quality (25 points):
           - GPA >= 3.5: 25 points
           - GPA >= 3.0: 20 points
           - GPA >= 2.5: 15 points
           - GPA < 2.5: 10 points
           - No grade info: 12 points (neutral)
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
                    required_level = 3  # Default to bachelor's if not specified
                
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
                    field_similarity = 0.8  # No specific field required
                
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
        """Normalize location string for comparison"""
        if not location_str:
            return ""
        return re.sub(r'[^\w\s]', '', location_str.lower().strip())
    
    def extract_resume_locations(self, resume) -> List[str]:
        """Extract all location information from resume"""
        locations = []
        
        # From contact info address
        if resume.contact_info.address:
            addr = resume.contact_info.address
            if addr.city:
                locations.append(addr.city)
            if addr.state:
                locations.append(addr.state)
            if addr.country:
                locations.append(addr.country)
        
        # From experience locations
        for exp in resume.experience:
            if exp.location:
                locations.append(exp.location)
        
        # From education locations
        for edu in resume.education:
            if edu.location:
                locations.append(edu.location)
        
        return [self.normalize_location(loc) for loc in locations if loc]
    
    def check_location_match(self, resume_locations: List[str], job_location: str) -> bool:
        """Check if any resume location matches job location"""
        job_loc_norm = self.normalize_location(job_location)
        
        for resume_loc in resume_locations:
            if resume_loc in job_loc_norm or job_loc_norm in resume_loc:
                return True
        
        return False
    
    def match_location(self, resume, job_description) -> MatchScore:
        """
        Match location requirements between resume and job description
        
        Location Matching Rules:
        1. Remote work (100 points):
           - Job is remote: Full score regardless of resume location
        
        2. Hybrid work (90 points):
           - Treated as onsite for matching purposes
           - Location match required
        
        3. Onsite work (100 points):
           - Must have location match for full score
           - No location in resume: 30 points penalty
           - No location in JD: No penalty
           - No location in both: No penalty
        
        4. Location match scoring:
           - Exact match: Full score
           - No match but both have locations: 20 points penalty
           - Resume has no location info: 30 points penalty
        """
        
        max_score = 100.0
        
        # Check remote work
        if job_description.is_remote:
            return MatchScore(max_score, max_score, "Remote work - location not relevant")
        
        # Extract resume locations
        resume_locations = self.extract_resume_locations(resume)
        
        # Get job locations
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
        
        # No location requirements in job
        if not job_locations:
            return MatchScore(max_score, max_score, "No specific location requirements")
        
        # No location info in resume - penalty for onsite/hybrid
        if not resume_locations:
            penalty_score = max_score - 30
            return MatchScore(penalty_score, max_score, 
                            "No location information in resume - 30 point penalty")
        
        # Check for location matches
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
    
    def __init__(self, model):
        self.model = model
        self.relevance_threshold = 0.5  # Minimum similarity for relevant experience
    
    def embed_text(self, text: str):
        """Generate embeddings for text using the sentence transformer model"""
        return self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using embeddings"""
        try:
            vec1 = self.embed_text(text1)
            vec2 = self.embed_text(text2)
            # Cosine similarity using normalized embeddings
            similarity = float(vec1.dot(vec2))
            return max(0.0, min(1.0, similarity))  # Clamp between 0 and 1
        except Exception:
            return 0.0
    
    def normalize_job_title(self, title: str) -> str:
        """Normalize job title for better matching"""
        if not title:
            return ""
        
        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^\w\s]', ' ', title.lower())
        # Remove common words that don't add semantic value
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in normalized.split() if word not in stop_words]
        return ' '.join(words).strip()
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats to datetime object"""
        if not date_str:
            return None
        
        try:
            # Handle "Present", "Current", etc.
            if date_str.lower() in ['present', 'current', 'now', 'ongoing']:
                return datetime.now()
            
            # Try to parse the date
            return parser.parse(date_str, fuzzy=True)
        except:
            return None
    
    def calculate_experience_duration(self, start_date: str, end_date: str) -> float:
        """Calculate experience duration in years"""
        start = self.parse_date(start_date)
        end = self.parse_date(end_date)
        
        if not start:
            return 0.0
        
        if not end:
            end = datetime.now()
        
        # Calculate difference in years
        duration = (end - start).days / 365.25
        return max(0.0, duration)
    
    def calculate_job_title_similarity(self, job_title: str, resume_experiences: List) -> List[Tuple[int, float, float]]:
        """
        Calculate similarity between job title and all resume experiences
        Returns: List of (experience_index, similarity_score, years_of_experience)
        """
        job_title_norm = self.normalize_job_title(job_title)
        similarities = []
        
        for i, exp in enumerate(resume_experiences):
            exp_title_norm = self.normalize_job_title(exp.job_title)
            
            # Calculate semantic similarity
            similarity = self.calculate_semantic_similarity(job_title_norm, exp_title_norm)
            
            # Calculate years of experience for this job
            years = self.calculate_experience_duration(exp.start_date, exp.end_date)
            
            similarities.append((i, similarity, years))
        
        return similarities
    
    def aggregate_relevant_experience(self, similarities: List[Tuple[int, float, float]]) -> Tuple[float, List[Tuple[int, float]]]:
        """
        Aggregate only relevant experience based on relevance threshold
        Returns: (relevant_years, relevant_jobs_details)
        """
        relevant_years = 0.0
        relevant_jobs = []
        
        for exp_idx, similarity, years in similarities:
            if similarity >= self.relevance_threshold:
                relevant_years += years
                relevant_jobs.append((exp_idx, similarity))
        
        return relevant_years, relevant_jobs
    
    def calculate_experience_score(self, relevant_years: float, required_min: float, required_max: Optional[float] = None) -> Tuple[float, str]:
        """
        Calculate experience score based only on relevant experience
        """
        
        if required_max is None:
            required_max = required_min * 1.5

        max_score = 100.0

        # Calculate score based only on relevant experience
        if relevant_years >= required_min:
            if relevant_years >= required_max:
                final_score = max_score
            else:
                # Scale between 60-100 based on progress from min to max
                progress = (relevant_years - required_min) / (required_max - required_min)
                final_score = 60.0 + (progress * 40.0)
        else:
            # Scale from 0-60 based on progress toward minimum
            if required_min > 0:
                final_score = (relevant_years / required_min) * 60.0
            else:
                final_score = 60.0

        final_score = min(max_score, final_score)

        # Explanation
        explanation_parts = []
        explanation_parts.append(f"Relevant experience: {relevant_years:.1f} years")
        explanation_parts.append(f"Required: {required_min:.1f}-{required_max:.1f} years")

        if relevant_years >= required_min:
            explanation_parts.append("✓ Meets minimum requirement")
        else:
            shortage = required_min - relevant_years
            explanation_parts.append(f"⚠ {shortage:.1f} years short of minimum")

        explanation = "; ".join(explanation_parts)
        
        return final_score, explanation

    def match_experience(self, resume_experiences: List, job_description) -> MatchScore:
        """
        Match experience requirements between resume and job description
        
        Experience Matching Rules:
        1. Job Title Similarity (Primary factor):
           - Uses semantic similarity with threshold of 0.5
           - Only experiences above threshold count as "relevant"
        
        2. Years Calculation:
           - Only relevant years are considered
           - Non-relevant experience is completely ignored
        
        3. Scoring (100 points total):
           - Based entirely on relevant experience
           - 0-60 points: Progress toward minimum requirement
           - 60-100 points: Progress from minimum to maximum requirement
        
        4. Requirements Matching:
           - Compares against min/max experience requirements
           - Penalties for not meeting minimum requirements
           - Bonuses for exceeding requirements
        """
        
        if not resume_experiences:
            return MatchScore(0.0, 100.0, "No work experience found in resume")
        
        # Get experience requirements from job description
        required_min = 0.0
        required_max = None
        
        if (job_description.qualifications and 
            job_description.qualifications.experience_years):
            exp_req = job_description.qualifications.experience_years
            required_min = exp_req.min or 0.0
            required_max = exp_req.max
        
        # If no specific requirements, use job level as indicator
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
        
        # Default minimum if nothing specified
        if required_min == 0.0:
            required_min = 1.0  # Default to 1 year minimum
        
        # Calculate similarities with job title
        job_title = job_description.title
        similarities = self.calculate_job_title_similarity(job_title, resume_experiences)
        
        # Aggregate only relevant experience
        relevant_years, relevant_jobs = self.aggregate_relevant_experience(similarities)
        
        # Calculate final score based only on relevant experience
        score, explanation = self.calculate_experience_score(
            relevant_years, required_min, required_max
        )
        
        # Add details about relevant jobs to explanation
        if relevant_jobs:
            job_details = []
            for exp_idx, similarity in relevant_jobs[:3]:  # Show top 3 relevant jobs
                exp = resume_experiences[exp_idx]
                job_details.append(f"{exp.job_title} ({similarity:.2f} similarity)")
            
            if job_details:
                explanation += f"; Relevant roles: {', '.join(job_details)}"
        elif relevant_years == 0:
            explanation += "; No relevant experience found"
        
        return MatchScore(score, 100.0, explanation)

class ConstraintMatcher:
    """Main class that combines all matching algorithms"""
    
    def __init__(self, model):
        self.model = model
        self.education_matcher = EducationMatcher(self.model)
        self.location_matcher = LocationMatcher()
        self.experience_matcher = ExperienceMatcher(self.model)
    
    def calculate_overall_score(self, resume, job_description, weights: Dict[str, float] = None) -> Dict:
        """
        Calculate overall matching score between resume and job description
        
        Updated weights:
        - Education: 0.25 (25%)
        - Location: 0.25 (25%)
        - Experience: 0.35 (35%) - Now implemented
        """
        
        if weights is None:
            weights = {
                'education': 0.35,
                'location': 0.30,
                'experience': 0.35,  
                'skills' : 0.0 # not implemented yet
            }
        
        results = {}
        
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
        
        # Experience matching - Now implemented!
        experience_score = self.experience_matcher.match_experience(resume.experience, job_description)
        results['experience'] = {
            'score': experience_score.score,
            'max_score': experience_score.max_score,
            'normalized_score': experience_score.normalized_score,
            'explanation': experience_score.explanation,
            'weight': weights['experience']
        }
        
        # Placeholder for skills (reduced weight)
        results['skills'] = {
            'score': 60.0, 'max_score': 100.0, 'normalized_score': 0.6,
            'explanation': 'Skills matching not implemented yet',
            'weight': weights['skills']
        }
        
        # Calculate weighted overall score
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
