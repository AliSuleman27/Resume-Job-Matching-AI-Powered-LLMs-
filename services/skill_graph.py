"""
Graph-based skill matching engine.

Loads two knowledge graphs (MIND Tech Ontology + ESCO Digital Skills hierarchy)
and provides multi-tier matching that replaces raw embedding cosine similarity.

Tiers:
  1 - Exact:       same canonical node after normalization     → 1.00
  2 - Synonym:     alias resolution to same node               → 0.95
  3a - Implies:    candidate skill implies JD skill (MIND)     → 0.85
  3b - Parent:     candidate has child, JD wants broader       → 0.75
  3c - Child:      candidate has parent, JD wants child        → 0.40
  4 - Related:     same domain/concept group                   → 0.30
  5 - Embedding:   fallback for unknown skills, capped         ≤ 0.55
"""

import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class SkillNode:
    canonical_id: str
    display_name: str
    synonyms: Set[str] = field(default_factory=set)
    implies: Set[str] = field(default_factory=set)      # canonical_ids
    concepts: Set[str] = field(default_factory=set)      # conceptual aspects
    domains: Set[str] = field(default_factory=set)       # e.g. Frontend, Backend
    broader: Optional[str] = None                         # parent canonical_id
    narrower: Set[str] = field(default_factory=set)      # child canonical_ids


@dataclass
class SkillMatchDetail:
    jd_skill: str
    candidate_skill: str
    similarity: float
    tier: str            # "exact", "synonym", "implies", "parent", "child", "related", "embedding"


@dataclass
class SkillsMatchSummary:
    composite_score: float                   # 0-1
    matched_details: List[SkillMatchDetail]
    missing_mandatory: List[str]
    missing_optional: List[str]
    extra_skills: List[str]
    mandatory_rate: float
    optional_rate: float


# ---------------------------------------------------------------------------
#  Normalization
# ---------------------------------------------------------------------------

# Patterns compiled once
_DOT_JS = re.compile(r'\.js\b', re.IGNORECASE)
_DOT_TS = re.compile(r'\.ts\b', re.IGNORECASE)
_NON_ALNUM = re.compile(r'[^a-z0-9+#/ ]')
_MULTI_SPACE = re.compile(r'\s+')


def normalize(raw: str) -> str:
    """Normalize a skill name for graph lookup.

    "React.js" → "reactjs", "C++" → "c++", "Vue.js" → "vuejs",
    "Node.js" → "nodejs", "ASP.NET" → "aspnet"
    """
    s = raw.strip().lower()
    # Collapse .js / .ts suffixes into the base word
    s = _DOT_JS.sub('js', s)
    s = _DOT_TS.sub('ts', s)
    # Replace .net → net (for ASP.NET, .NET)
    s = s.replace('.net', 'net').replace('.', '')
    # Keep +, #, / as they are meaningful (C++, C#, CI/CD)
    s = _NON_ALNUM.sub(' ', s)
    s = _MULTI_SPACE.sub(' ', s).strip()
    # Remove spaces for compact matching ("react js" → "reactjs")
    # but keep multi-word phrases readable for longer names
    if len(s.split()) <= 3:
        s = s.replace(' ', '')
    return s


# ---------------------------------------------------------------------------
#  SkillGraph
# ---------------------------------------------------------------------------

class SkillGraph:
    def __init__(self):
        self._nodes: Dict[str, SkillNode] = {}           # canonical_id → SkillNode
        self._alias_index: Dict[str, str] = {}           # normalized form → canonical_id
        self._domain_groups: Dict[str, Set[str]] = {}    # domain → set of canonical_ids
        self._concept_groups: Dict[str, Set[str]] = {}   # concept → set of canonical_ids
        self._implies_cache: Dict[str, Set[str]] = {}    # cached transitive implies
        self._ancestors_cache: Dict[str, Set[str]] = {}  # cached transitive broader
        self._built = False

    def _ensure_built(self):
        if not self._built:
            self._build()
            self._built = True

    # ------------------------------------------------------------------
    #  Graph construction
    # ------------------------------------------------------------------

    def _get_or_create_node(self, name: str) -> str:
        """Get or create a node, returning its canonical_id."""
        cid = normalize(name)
        if cid not in self._nodes:
            self._nodes[cid] = SkillNode(canonical_id=cid, display_name=name)
        return cid

    def _register_alias(self, alias: str, canonical_id: str):
        """Register a normalized alias → canonical_id mapping."""
        norm = normalize(alias)
        if norm and norm not in self._alias_index:
            self._alias_index[norm] = canonical_id

    def _build(self):
        """Load all data sources and build the graph."""
        logger.info("Building skill graph...")
        self._load_mind_skills()
        self._load_esco_hierarchy()
        self._load_tech_skills_taxonomy()
        self._build_group_indexes()
        logger.info(
            "Skill graph built: %d nodes, %d aliases, %d domain groups, %d concept groups",
            len(self._nodes), len(self._alias_index),
            len(self._domain_groups), len(self._concept_groups),
        )

    def _load_mind_skills(self):
        """Load MIND Tech Ontology skills."""
        path = _DATA_DIR / "mind_skills.json"
        if not path.exists():
            logger.warning("MIND skills file not found at %s", path)
            return

        with open(path, encoding="utf-8") as f:
            skills = json.load(f)

        for entry in skills:
            name = entry.get("name", "").strip()
            if not name:
                continue

            cid = self._get_or_create_node(name)
            node = self._nodes[cid]

            # Register all synonyms
            self._register_alias(name, cid)
            for syn in entry.get("synonyms", []):
                syn = syn.strip()
                if syn:
                    node.synonyms.add(syn)
                    self._register_alias(syn, cid)

            # Store implication references (resolve later)
            for implied_name in entry.get("impliesKnowingSkills", []):
                implied_name = implied_name.strip()
                if implied_name:
                    implied_cid = self._get_or_create_node(implied_name)
                    node.implies.add(implied_cid)

            # Store concepts and domains
            for concept in entry.get("conceptualAspects", []):
                node.concepts.add(concept)
            for domain in entry.get("domains", []):
                node.domains.add(domain)

        logger.info("Loaded %d MIND skills", len(skills))

    def _load_esco_hierarchy(self):
        """Load ESCO digital skills hierarchy CSV."""
        path = _DATA_DIR / "esco_digital_skills.csv"
        if not path.exists():
            logger.warning("ESCO skills file not found at %s", path)
            return

        count = 0
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label", "").strip()
                if not label:
                    continue

                cid = self._get_or_create_node(label)
                node = self._nodes[cid]
                self._register_alias(label, cid)

                # Alt labels as synonyms
                alt_labels = row.get("alt_labels", "")
                if alt_labels:
                    for alt in alt_labels.split("|"):
                        alt = alt.strip()
                        if alt:
                            node.synonyms.add(alt)
                            self._register_alias(alt, cid)

                # Broader (parent) relationship
                broader = row.get("broader_label", "").strip()
                if broader:
                    broader_cid = self._get_or_create_node(broader)
                    node.broader = broader_cid
                    self._nodes[broader_cid].narrower.add(cid)

                count += 1

        logger.info("Loaded %d ESCO hierarchy entries", count)

    def _load_tech_skills_taxonomy(self):
        """Import TECH_SKILLS from resume_post_processor as basic nodes."""
        try:
            from services.resume_post_processor import TECH_SKILLS
        except ImportError:
            logger.warning("Could not import TECH_SKILLS from resume_post_processor")
            return

        added = 0
        for skill_lower, category in TECH_SKILLS.items():
            norm = normalize(skill_lower)
            if norm not in self._alias_index:
                # New skill not yet in graph
                cid = self._get_or_create_node(skill_lower)
                self._register_alias(skill_lower, cid)
                self._nodes[cid].domains.add(category)
                added += 1
            else:
                # Already exists — add domain if missing
                existing_cid = self._alias_index[norm]
                self._nodes[existing_cid].domains.add(category)

        logger.info("Imported TECH_SKILLS taxonomy: %d new nodes added", added)

    def _build_group_indexes(self):
        """Build domain and concept group indexes for Tier 4 matching."""
        for cid, node in self._nodes.items():
            for domain in node.domains:
                self._domain_groups.setdefault(domain, set()).add(cid)
            for concept in node.concepts:
                self._concept_groups.setdefault(concept, set()).add(cid)

    # ------------------------------------------------------------------
    #  Traversal helpers (with caching)
    # ------------------------------------------------------------------

    def _get_all_implied(self, cid: str, depth: int = 0) -> Set[str]:
        """Get all skills transitively implied by this skill (max depth 5)."""
        if cid in self._implies_cache:
            return self._implies_cache[cid]

        if depth > 5 or cid not in self._nodes:
            return set()

        result = set()
        for implied_cid in self._nodes[cid].implies:
            result.add(implied_cid)
            result |= self._get_all_implied(implied_cid, depth + 1)

        self._implies_cache[cid] = result
        return result

    def _get_all_ancestors(self, cid: str, depth: int = 0) -> Set[str]:
        """Get all broader (ancestor) skills via hierarchy (max depth 5)."""
        if cid in self._ancestors_cache:
            return self._ancestors_cache[cid]

        if depth > 5 or cid not in self._nodes:
            return set()

        result = set()
        broader = self._nodes[cid].broader
        if broader:
            result.add(broader)
            result |= self._get_all_ancestors(broader, depth + 1)

        self._ancestors_cache[cid] = result
        return result

    def _get_all_descendants(self, cid: str, depth: int = 0) -> Set[str]:
        """Get all narrower (descendant) skills via hierarchy (max depth 3)."""
        if depth > 3 or cid not in self._nodes:
            return set()

        result = set()
        for child_cid in self._nodes[cid].narrower:
            result.add(child_cid)
            result |= self._get_all_descendants(child_cid, depth + 1)

        return result

    def _share_domain_or_concept(self, cid1: str, cid2: str) -> bool:
        """Check if two skills share a domain or concept group."""
        n1 = self._nodes.get(cid1)
        n2 = self._nodes.get(cid2)
        if not n1 or not n2:
            return False

        # Shared domain
        if n1.domains & n2.domains:
            return True

        # Shared concept
        if n1.concepts & n2.concepts:
            return True

        # ESCO siblings (same parent)
        if n1.broader and n1.broader == n2.broader:
            return True

        return False

    # ------------------------------------------------------------------
    #  Resolution
    # ------------------------------------------------------------------

    def _resolve(self, raw: str) -> Optional[str]:
        """Resolve a raw skill string to a canonical_id, or None."""
        self._ensure_built()
        norm = normalize(raw)
        return self._alias_index.get(norm)

    # ------------------------------------------------------------------
    #  Single-pair matching
    # ------------------------------------------------------------------

    def match_skill(self, jd_skill: str, cand_skill: str) -> float:
        """Multi-tier skill matching.

        Returns a score 0.0-1.0 for known relationships, or -1.0 as a
        sentinel when neither skill is in the graph (caller should use
        embedding fallback capped at 0.55).
        """
        self._ensure_built()

        jd_norm = normalize(jd_skill)
        cand_norm = normalize(cand_skill)

        # Tier 1 — Exact normalization match
        if jd_norm == cand_norm:
            return 1.00

        jd_cid = self._alias_index.get(jd_norm)
        cand_cid = self._alias_index.get(cand_norm)

        # If both are unknown, signal embedding fallback
        if jd_cid is None and cand_cid is None:
            return -1.0

        # If only one is known, can't compare via graph
        if jd_cid is None or cand_cid is None:
            return -1.0

        # Tier 2 — Synonym (both resolve to same node)
        if jd_cid == cand_cid:
            return 0.95

        # Tier 3a — Candidate implies JD skill
        if jd_cid in self._get_all_implied(cand_cid):
            return 0.85

        # Tier 3b — Candidate has child skill, JD wants broader parent
        #           (candidate is more specific → still good)
        cand_ancestors = self._get_all_ancestors(cand_cid)
        if jd_cid in cand_ancestors:
            return 0.75

        # Tier 3c — Candidate has parent, JD wants specific child
        #           (candidate is more general → weak match)
        jd_ancestors = self._get_all_ancestors(jd_cid)
        if cand_cid in jd_ancestors:
            return 0.40

        # Also check: candidate implies a parent of JD skill (partial chain)
        cand_implied = self._get_all_implied(cand_cid)
        if cand_implied & jd_ancestors:
            return 0.60

        # And: JD skill implies candidate's parent
        jd_implied = self._get_all_implied(jd_cid)
        if cand_cid in jd_implied:
            return 0.50

        # Tier 4 — Related (same domain/concept group or ESCO siblings)
        if self._share_domain_or_concept(jd_cid, cand_cid):
            return 0.30

        # Known but unrelated
        return 0.10

    # ------------------------------------------------------------------
    #  Batch matching
    # ------------------------------------------------------------------

    def match_skills_batch(
        self,
        jd_mandatory: List[str],
        jd_optional: List[str],
        jd_tools: List[str],
        candidate_skills: List[str],
    ) -> SkillsMatchSummary:
        """Match JD skills against candidate skills using graph + embedding fallback.

        Returns a SkillsMatchSummary with composite score, per-skill details,
        and lists of missing/extra skills.
        """
        self._ensure_built()

        all_jd = jd_mandatory + jd_optional + jd_tools
        if not all_jd:
            return SkillsMatchSummary(
                composite_score=1.0, matched_details=[], missing_mandatory=[],
                missing_optional=[], extra_skills=list(candidate_skills),
                mandatory_rate=1.0, optional_rate=1.0,
            )
        if not candidate_skills:
            return SkillsMatchSummary(
                composite_score=0.0, matched_details=[], missing_mandatory=list(jd_mandatory),
                missing_optional=list(jd_optional + jd_tools), extra_skills=[],
                mandatory_rate=0.0, optional_rate=0.0,
            )

        mandatory_set = set(jd_mandatory)

        # First pass: identify which skill pairs need embedding fallback
        needs_embedding: Set[str] = set()
        graph_scores: Dict[str, Dict[str, float]] = {}
        for jd_skill in all_jd:
            graph_scores[jd_skill] = {}
            for cand_skill in candidate_skills:
                score = self.match_skill(jd_skill, cand_skill)
                graph_scores[jd_skill][cand_skill] = score
                if score == -1.0:
                    needs_embedding.add(jd_skill.lower())
                    needs_embedding.add(cand_skill.lower())

        # Pre-compute embeddings ONCE per unique skill (not per pair)
        emb_vectors: Dict[str, list] = {}
        if needs_embedding:
            emb_vectors = self._batch_get_embeddings(list(needs_embedding))

        # Build score matrix with graph scores + embedding fallback
        score_matrix: Dict[str, Dict[str, Tuple[float, str]]] = {}
        for jd_skill in all_jd:
            score_matrix[jd_skill] = {}
            for cand_skill in candidate_skills:
                graph_score = graph_scores[jd_skill][cand_skill]
                if graph_score == -1.0:
                    # Embedding fallback using pre-computed vectors
                    emb_score = self._cosine_from_cache(
                        jd_skill.lower(), cand_skill.lower(), emb_vectors
                    )
                    score_matrix[jd_skill][cand_skill] = (emb_score, "embedding")
                elif graph_score >= 0.95:
                    tier = "exact" if graph_score == 1.0 else "synonym"
                    score_matrix[jd_skill][cand_skill] = (graph_score, tier)
                elif graph_score >= 0.85:
                    score_matrix[jd_skill][cand_skill] = (graph_score, "implies")
                elif graph_score >= 0.75:
                    score_matrix[jd_skill][cand_skill] = (graph_score, "parent")
                elif graph_score >= 0.50:
                    score_matrix[jd_skill][cand_skill] = (graph_score, "implies_partial")
                elif graph_score >= 0.40:
                    score_matrix[jd_skill][cand_skill] = (graph_score, "child")
                elif graph_score >= 0.30:
                    score_matrix[jd_skill][cand_skill] = (graph_score, "related")
                else:
                    score_matrix[jd_skill][cand_skill] = (graph_score, "weak")

        # Greedy 1-to-1 matching: mandatory first, then optional
        used_cands: Set[str] = set()
        matched_details: List[SkillMatchDetail] = []
        jd_matched: Set[str] = set()

        # Build all (jd_skill, cand_skill, score, tier) tuples, sort by score desc
        all_pairs = []
        for jd_skill in all_jd:
            for cand_skill in candidate_skills:
                score, tier = score_matrix[jd_skill][cand_skill]
                is_mandatory = jd_skill in mandatory_set
                all_pairs.append((jd_skill, cand_skill, score, tier, is_mandatory))

        # Sort: mandatory first, then by score descending
        all_pairs.sort(key=lambda x: (-x[4], -x[2]))

        for jd_skill, cand_skill, score, tier, _ in all_pairs:
            if jd_skill in jd_matched or cand_skill in used_cands:
                continue
            # Minimum threshold: don't match with very weak scores
            if score < 0.25:
                continue
            jd_matched.add(jd_skill)
            used_cands.add(cand_skill)
            matched_details.append(SkillMatchDetail(
                jd_skill=jd_skill,
                candidate_skill=cand_skill,
                similarity=round(score, 3),
                tier=tier,
            ))

        # Compute quality scores (tier-weighted, not binary)
        mandatory_total = 0.0
        mandatory_count = len(jd_mandatory) if jd_mandatory else 0
        optional_total = 0.0
        optional_count = len(jd_optional) + len(jd_tools)

        for detail in matched_details:
            if detail.jd_skill in mandatory_set:
                mandatory_total += detail.similarity
            else:
                optional_total += detail.similarity

        mandatory_rate = (mandatory_total / mandatory_count) if mandatory_count > 0 else 1.0
        optional_rate = (optional_total / optional_count) if optional_count > 0 else 1.0

        composite = mandatory_rate * 0.75 + optional_rate * 0.25

        # Gather missing and extra
        missing_mandatory = [s for s in jd_mandatory if s not in jd_matched]
        missing_optional = [s for s in (jd_optional + jd_tools) if s not in jd_matched]
        extra_skills = [s for s in candidate_skills if s not in used_cands]

        return SkillsMatchSummary(
            composite_score=round(composite, 4),
            matched_details=matched_details,
            missing_mandatory=missing_mandatory,
            missing_optional=missing_optional,
            extra_skills=extra_skills,
            mandatory_rate=round(mandatory_rate, 4),
            optional_rate=round(optional_rate, 4),
        )

    # ------------------------------------------------------------------
    #  Embedding fallback (batch pre-compute, capped at 0.55)
    # ------------------------------------------------------------------

    def _batch_get_embeddings(self, skills: List[str]) -> Dict[str, list]:
        """Pre-compute embeddings for a batch of skill strings.

        Calls get_embedding() once per unique skill (not per pair).
        Results are cached by embedding_service's MongoDB cache, so
        subsequent calls for the same skill are instant.
        """
        vectors: Dict[str, list] = {}
        if not skills:
            return vectors
        try:
            from services.embedding_service import get_embedding
            for skill in skills:
                if skill not in vectors:
                    vectors[skill] = get_embedding(skill)
        except Exception as e:
            logger.debug("Batch embedding error: %s", e)
        return vectors

    def _cosine_from_cache(self, skill_a: str, skill_b: str,
                           vectors: Dict[str, list]) -> float:
        """Cosine similarity from pre-computed vectors, capped at 0.55."""
        vec_a = vectors.get(skill_a)
        vec_b = vectors.get(skill_b)
        if vec_a is None or vec_b is None:
            return 0.0
        try:
            from services.embedding_service import cosine_similarity
            raw = cosine_similarity(vec_a, vec_b)
            return min(0.55, max(0.0, raw))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
#  Singleton
# ---------------------------------------------------------------------------

_graph_instance: Optional[SkillGraph] = None


def get_skill_graph() -> SkillGraph:
    """Return the singleton SkillGraph (lazy-initialized)."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = SkillGraph()
    return _graph_instance
