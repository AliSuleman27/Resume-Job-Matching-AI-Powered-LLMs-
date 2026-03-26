"""
Test script for Skill Graph matching.

Run:  python test_skill_graph.py

Verifies the MIND + ESCO skill graph loads and produces correct tier-based
matching for common job-description scenarios.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
os.environ['SKIP_EMBEDDING'] = '1'

# Patch embedding service so we can run without HF token
import services.embedding_service as es
es.get_embedding = lambda x: [0.0] * 768
es.cosine_similarity = lambda a, b: 0.0

from services.skill_graph import get_skill_graph


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def test_graph_loading():
    print_header("TEST 1: Graph Loading")
    sg = get_skill_graph()
    sg._ensure_built()
    print(f"  Nodes loaded:   {len(sg._nodes):,}")
    print(f"  Aliases:        {len(sg._alias_index):,}")
    print(f"  Domain groups:  {len(sg._domain_groups)}")
    print(f"  Concept groups: {len(sg._concept_groups)}")
    assert len(sg._nodes) > 3000, "Expected 3000+ nodes from MIND"
    assert len(sg._alias_index) > 5000, "Expected 5000+ aliases"
    print("  PASSED")
    return sg


def test_tier_matching(sg):
    print_header("TEST 2: Multi-Tier Matching (match_skill)")

    # (jd_skill, cand_skill, expected_score, description)
    cases = [
        # Tier 1 - Exact
        ("Python",             "Python",             1.00, "Exact same skill"),
        ("Docker",             "Docker",             1.00, "Exact same skill"),

        # Tier 2 - Synonym (alias resolution)
        ("React",              "React.js",           0.95, "MIND synonym: React / React.js"),
        ("React",              "ReactJS",            0.95, "MIND synonym: React / ReactJS"),
        ("Kubernetes",         "K8s",                0.95, "MIND synonym: Kubernetes / K8s"),
        ("Machine Learning",   "ML",                 0.95, "ESCO alt_label: Machine Learning / ML"),
        ("PostgreSQL",         "Postgres",           0.95, "MIND synonym: PostgreSQL / Postgres"),
        ("Vue.js",             "Vue",                0.95, "MIND synonym: Vue.js / Vue"),
        ("TypeScript",         "TS",                 0.95, "MIND synonym: TypeScript / TS"),
        ("AWS",                "Amazon Web Services", 0.95, "MIND synonym: AWS / Amazon Web Services"),
        ("Natural Language Processing", "NLP",       0.95, "ESCO alt_label"),
        ("Google Cloud Platform", "GCP",             0.95, "ESCO alt_label"),

        # Tier 3a - Implies (candidate skill implies JD skill)
        ("React",              "Next.js",            0.85, "Next.js implies React (MIND)"),
        ("JavaScript",         "TypeScript",         0.85, "TypeScript implies JavaScript (MIND)"),
        ("Python",             "Flask",              0.85, "Flask implies Python (MIND)"),
        ("Python",             "Django",             0.85, "Django implies Python (MIND)"),
        ("Docker",             "Kubernetes",         0.85, "Kubernetes implies Docker (MIND)"),
        ("SQL",                "PostgreSQL",         0.85, "PostgreSQL implies SQL (MIND)"),
        ("HTML",               "React",              0.85, "React implies HTML (MIND)"),
        ("CSS",                "Angular",            0.85, "Angular implies CSS (MIND)"),
        ("JavaScript",         "Angular",            0.85, "Angular implies JavaScript (MIND)"),

        # Tier 3b - Parent (candidate has child, JD wants broader)
        ("NoSQL Databases",    "MongoDB",            0.75, "MongoDB is child of NoSQL (ESCO)"),
        ("Frontend Frameworks","React",              0.75, "React is child of Frontend Frameworks (ESCO)"),
        ("Relational Databases","PostgreSQL",        0.75, "PostgreSQL under Relational DBs (ESCO)"),
        ("Cloud Platforms",    "AWS",                0.75, "AWS under Cloud Platforms (ESCO)"),

        # Tier 3c - Child (candidate has parent, JD wants child)
        ("MongoDB",            "NoSQL Databases",    0.40, "Candidate has parent, JD wants child"),
        ("React",              "Frontend Frameworks", 0.40, "Candidate has parent, JD wants child"),

        # Tier 4 - Related (same domain/concept group)
        ("React",              "Vue.js",             0.30, "Both Frontend frameworks"),
        ("React",              "Angular",            0.30, "Both Frontend frameworks"),
        ("Python",             "Java",               0.30, "Both programming languages"),
        ("PostgreSQL",         "MySQL",              0.30, "Both relational DBs"),
        ("MongoDB",            "Redis",              0.30, "Both NoSQL databases"),

        # Unrelated
        ("React",              "PostgreSQL",         0.10, "Frontend vs Database = unrelated"),
    ]

    passed = 0
    failed = 0
    for jd, cand, expected, desc in cases:
        score = sg.match_skill(jd, cand)
        ok = abs(score - expected) < 0.02
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status}  {jd:30s} <- {cand:25s} = {score:5.2f} (expect {expected:.2f}) [{desc}]")

    print(f"\n  Results: {passed} passed, {failed} failed out of {len(cases)}")
    return failed == 0


def test_batch_matching(sg):
    print_header("TEST 3: Batch Matching (match_skills_batch)")

    # Scenario: Full-stack developer JD vs candidate with partial match
    result = sg.match_skills_batch(
        jd_mandatory=["React", "TypeScript", "Node.js", "PostgreSQL"],
        jd_optional=["Docker", "AWS", "Redis"],
        jd_tools=["Git", "Jira"],
        candidate_skills=[
            "React.js", "Next.js", "JavaScript", "TypeScript",
            "Postgres", "Docker", "K8s", "GitHub",
            "Python", "Flask", "Machine Learning"
        ]
    )

    print(f"  Composite score:  {result.composite_score:.3f}")
    print(f"  Mandatory rate:   {result.mandatory_rate:.3f}")
    print(f"  Optional rate:    {result.optional_rate:.3f}")
    print()
    print(f"  {'JD Skill':25s} {'Candidate Skill':25s} {'Score':>6s}  {'Tier':15s}")
    print(f"  {'-'*25} {'-'*25} {'-'*6}  {'-'*15}")
    for d in result.matched_details:
        print(f"  {d.jd_skill:25s} {d.candidate_skill:25s} {d.similarity:5.2f}  {d.tier}")

    print(f"\n  Missing mandatory: {result.missing_mandatory}")
    print(f"  Missing optional:  {result.missing_optional}")
    print(f"  Extra skills:      {result.extra_skills}")

    # Verify key expectations
    matched_jd = {d.jd_skill for d in result.matched_details}
    assert "React" in matched_jd, "React should be matched"
    assert "TypeScript" in matched_jd, "TypeScript should be matched"
    assert "Docker" in matched_jd, "Docker should be matched"

    # Verify React matched to React.js (synonym) not Next.js
    react_match = next(d for d in result.matched_details if d.jd_skill == "React")
    assert react_match.candidate_skill == "React.js", f"React should match React.js, got {react_match.candidate_skill}"
    assert react_match.tier == "synonym", f"React-React.js should be synonym, got {react_match.tier}"

    # Verify TypeScript matched exactly
    ts_match = next(d for d in result.matched_details if d.jd_skill == "TypeScript")
    assert ts_match.candidate_skill == "TypeScript", f"TypeScript should match exactly"
    assert ts_match.tier == "exact", f"TypeScript should be exact, got {ts_match.tier}"

    print("\n  PASSED - Key assertions verified")
    return True


def test_false_positive_prevention(sg):
    print_header("TEST 4: False Positive Prevention")

    # These pairs should NOT score high (the old system had false positives here)
    false_positive_cases = [
        ("React",      "Vue.js",      "Different frontend frameworks"),
        ("Angular",    "React",       "Different frontend frameworks"),
        ("PostgreSQL", "MongoDB",     "Different DB types (SQL vs NoSQL)"),
        ("Python",     "JavaScript",  "Different programming languages"),
        ("Docker",     "Terraform",   "Different DevOps tools"),
        ("AWS",        "Azure",       "Different cloud providers"),
    ]

    all_ok = True
    for jd, cand, desc in false_positive_cases:
        score = sg.match_skill(jd, cand)
        # Should score <= 0.30 (related at most, NOT 0.70+ like old embedding system)
        ok = score <= 0.35
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {status}  {jd:20s} vs {cand:20s} = {score:.2f} (<= 0.35 required) [{desc}]")

    if all_ok:
        print("\n  PASSED - No false positives!")
    else:
        print("\n  FAILED - Some false positives detected")
    return all_ok


def test_false_negative_prevention(sg):
    print_header("TEST 5: False Negative Prevention")

    # These pairs should score high (the old system missed them due to low embedding similarity)
    false_negative_cases = [
        ("Kubernetes",         "K8s",                "Abbreviation synonym"),
        ("Machine Learning",   "ML",                 "Abbreviation synonym"),
        ("PostgreSQL",         "Postgres",           "Short form synonym"),
        ("Natural Language Processing", "NLP",        "Abbreviation synonym"),
        ("Google Cloud Platform", "GCP",             "Abbreviation synonym"),
        ("React",              "Next.js",            "Framework implication"),
        ("JavaScript",         "TypeScript",         "Language implication"),
        ("Python",             "Django",             "Framework implication"),
    ]

    all_ok = True
    for jd, cand, desc in false_negative_cases:
        score = sg.match_skill(jd, cand)
        # Should score >= 0.80
        ok = score >= 0.80
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {status}  {jd:30s} <- {cand:20s} = {score:.2f} (>= 0.80 required) [{desc}]")

    if all_ok:
        print("\n  PASSED - No false negatives!")
    else:
        print("\n  FAILED - Some matches were missed")
    return all_ok


def test_constraint_matcher_integration():
    print_header("TEST 6: ConstraintMatcher Integration")

    from services.constraint_matcher import SkillsMatcher

    class FakeSkills:
        def __init__(self):
            self.mandatory = ["React", "TypeScript", "Python"]
            self.optional = ["Docker", "AWS"]
            self.tools = ["Git"]

    class FakeResume:
        class Skill:
            def __init__(self, name): self.skill_name = name
        def __init__(self):
            self.skills = [self.Skill(s) for s in ["React.js", "TS", "Flask", "GitHub", "K8s"]]
            self.experience = []

    class FakeJD:
        def __init__(self):
            self.skills = FakeSkills()

    sm = SkillsMatcher()
    result = sm.match_skills(FakeResume(), FakeJD())

    print(f"  Score: {result.score:.1f}/100")
    print(f"  Explanation: {result.explanation}")
    print(f"  Has details: {bool(result.details)}")

    if result.details:
        print(f"\n  Matched skills:")
        for m in result.details.get('matched', []):
            mand = " (mandatory)" if m['is_mandatory'] else ""
            print(f"    {m['jd_skill']:20s} <- {m['candidate_skill']:20s} [{m['tier']}] {m['similarity']:.2f}{mand}")
        print(f"  Missing mandatory: {result.details.get('missing_mandatory', [])}")
        print(f"  Missing optional:  {result.details.get('missing_optional', [])}")
        print(f"  Extra skills:      {result.details.get('extra_skills', [])}")

    assert result.details, "MatchScore should have details"
    assert 'matched' in result.details, "Details should have 'matched'"
    assert any(m['tier'] in ('exact', 'synonym') for m in result.details['matched']), \
        "Should have at least one exact/synonym match"

    print("\n  PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  SKILL GRAPH TEST SUITE")
    print("  Verifying MIND Tech Ontology + ESCO hierarchy matching")
    print("="*70)

    sg = test_graph_loading()

    results = [
        ("Tier Matching",           test_tier_matching(sg)),
        ("Batch Matching",          test_batch_matching(sg)),
        ("False Positive Prevention", test_false_positive_prevention(sg)),
        ("False Negative Prevention", test_false_negative_prevention(sg)),
        ("ConstraintMatcher Integration", test_constraint_matcher_integration()),
    ]

    print_header("SUMMARY")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {status}  {name}")

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_passed else 1)
