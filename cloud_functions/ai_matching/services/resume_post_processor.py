"""
Deterministic post-processor for LLM-parsed resumes.

Fixes common LLM parsing failures:
1. Category headers extracted as skill names instead of individual skills
2. Missing skills that are clearly present in the raw text
3. Empty skills_used in experience entries
4. Missing summary
"""

import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill Taxonomy (~500 entries): { "lowercase_name": "Category" }
# ---------------------------------------------------------------------------
_PROGRAMMING_LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#", "Go",
    "Rust", "Ruby", "PHP", "Swift", "Kotlin", "R", "Scala", "Dart", "SQL",
    "Bash", "Shell", "Perl", "Lua", "Haskell", "Elixir", "Erlang", "Clojure",
    "Groovy", "MATLAB", "Julia", "Objective-C", "Assembly", "VHDL", "Verilog",
    "Fortran", "COBOL", "F#", "OCaml", "Zig", "Nim", "Solidity", "Move",
    "CoffeeScript", "Visual Basic", "VBA", "PowerShell", "Apex",
    "PL/SQL", "T-SQL", "HCL", "YAML", "JSON", "XML", "HTML", "CSS", "SASS",
    "SCSS", "Less", "GraphQL", "Prolog", "Lisp", "Scheme", "Racket",
]

_AI_ML = [
    "PyTorch", "TensorFlow", "Scikit-Learn", "Keras", "HuggingFace",
    "Hugging Face", "OpenCV", "NLTK", "spaCy", "LangChain", "LangGraph",
    "CrewAI", "FAISS", "Chroma", "ChromaDB", "Pinecone", "Weaviate",
    "Milvus", "Qdrant", "LlamaIndex", "AutoGen", "OpenAI", "GPT",
    "GPT-4", "GPT-3", "Claude", "Anthropic", "Gemini", "PaLM", "BERT",
    "RoBERTa", "XGBoost", "LightGBM", "CatBoost", "Random Forest",
    "SVM", "Neural Networks", "Deep Learning", "Machine Learning",
    "NLP", "Natural Language Processing", "Computer Vision",
    "Reinforcement Learning", "GANs", "Transformers", "Diffusion Models",
    "Stable Diffusion", "Midjourney", "DALL-E", "RAG",
    "Retrieval Augmented Generation", "Fine-tuning", "RLHF", "LoRA",
    "QLoRA", "PEFT", "MLflow", "Weights & Biases", "W&B", "WandB",
    "DVC", "Optuna", "Ray", "Dask", "PySpark", "Spark MLlib",
    "SageMaker", "Vertex AI", "Azure ML", "Bedrock", "Ollama",
    "vLLM", "TensorRT", "ONNX", "CoreML", "MediaPipe", "Detectron2",
    "YOLO", "YOLOv5", "YOLOv8", "Ultralytics", "MMDetection",
    "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn", "Plotly",
    "Bokeh", "Altair", "Statsmodels", "SymPy", "Gensim",
    "Sentence Transformers", "OpenAI API", "Groq", "Mistral",
    "LLM", "LLMs", "Large Language Models", "Prompt Engineering",
    "Vector Databases", "Embeddings", "Semantic Search", "AI Agents",
    "Multi-Agent Systems", "Agentic AI", "AutoML", "Feature Engineering",
    "Model Deployment", "Model Serving", "ML Ops", "MLOps",
    "Data Science", "Data Analysis", "Data Engineering",
    "ETL", "Data Pipeline", "Data Warehousing", "Data Modeling",
    "A/B Testing", "Statistical Analysis", "Bayesian",
    "Time Series", "Anomaly Detection", "Recommendation Systems",
    "Chatbot", "Conversational AI", "Speech Recognition",
    "Text-to-Speech", "OCR", "Image Classification",
    "Object Detection", "Image Segmentation", "Semantic Segmentation",
    "Named Entity Recognition", "NER", "Sentiment Analysis",
    "Text Classification", "Question Answering", "Summarization",
    "Translation", "Information Extraction", "Knowledge Graphs",
]

_WEB_FRAMEWORKS = [
    "React", "React.js", "ReactJS", "Angular", "AngularJS", "Vue.js",
    "Vue", "VueJS", "Next.js", "NextJS", "Nuxt.js", "NuxtJS",
    "Svelte", "SvelteKit", "Gatsby", "Remix", "Astro",
    "Flask", "Django", "FastAPI", "Express.js", "Express", "ExpressJS",
    "Spring Boot", "Spring", "Node.js", "NodeJS", "Deno", "Bun",
    "Ruby on Rails", "Rails", "Laravel", "Symfony", "CodeIgniter",
    "ASP.NET", ".NET", ".NET Core", "Blazor", "Gin", "Echo", "Fiber",
    "Chi", "Actix", "Rocket", "Axum", "Phoenix", "Hono",
    "jQuery", "Bootstrap", "Tailwind CSS", "Tailwind",
    "Material UI", "MUI", "Chakra UI", "Ant Design", "Shadcn",
    "Styled Components", "Emotion", "Redux", "MobX", "Zustand",
    "React Query", "TanStack Query", "SWR", "Apollo Client",
    "Prisma", "Drizzle", "Sequelize", "TypeORM", "SQLAlchemy",
    "Alembic", "Mongoose", "PyMongo", "Celery", "Gunicorn", "Uvicorn",
    "Nginx", "Apache", "Caddy", "REST", "REST API", "RESTful",
    "gRPC", "WebSocket", "WebSockets", "Socket.IO", "SSE",
    "Swagger", "OpenAPI", "Pydantic", "Marshmallow",
    "Jinja2", "EJS", "Handlebars", "Pug", "Thymeleaf",
    "Webpack", "Vite", "Rollup", "esbuild", "Parcel", "Turbopack",
    "Babel", "SWC", "ESLint", "Prettier", "Storybook",
    "Three.js", "D3.js", "Chart.js", "Leaflet", "Mapbox",
    "Electron", "Tauri", "React Native", "Flutter", "Ionic",
    "Expo", "SwiftUI", "Jetpack Compose", "Xamarin", "MAUI",
    "WordPress", "Shopify", "Magento", "Strapi", "Contentful",
    "Sanity", "Ghost", "Payload CMS", "KeystoneJS",
    "tRPC", "Hapi", "Koa", "NestJS", "AdonisJS",
    "HTMX", "Alpine.js", "Stimulus", "Turbo", "Hotwire",
]

_DATABASES = [
    "PostgreSQL", "Postgres", "MySQL", "MariaDB", "SQLite",
    "MongoDB", "Redis", "Memcached", "Firebase", "Firestore",
    "Supabase", "Elasticsearch", "OpenSearch", "Solr",
    "DynamoDB", "Cassandra", "CouchDB", "CouchBase", "Neo4j",
    "ArangoDB", "InfluxDB", "TimescaleDB", "ClickHouse",
    "BigQuery", "Redshift", "Snowflake", "Databricks",
    "Apache Hive", "Apache HBase", "Apache Kafka", "Kafka",
    "RabbitMQ", "ActiveMQ", "Apache Pulsar", "NATS",
    "Amazon RDS", "Amazon Aurora", "Azure SQL", "Cloud SQL",
    "PlanetScale", "Neon", "CockroachDB", "TiDB", "Vitess",
    "Oracle DB", "Oracle", "SQL Server", "MSSQL",
    "Microsoft SQL Server", "IBM Db2",
]

_CLOUD_DEVOPS = [
    "AWS", "Amazon Web Services", "GCP", "Google Cloud Platform",
    "Google Cloud", "Azure", "Microsoft Azure", "DigitalOcean",
    "Heroku", "Vercel", "Netlify", "Cloudflare", "Render",
    "Railway", "Fly.io", "Linode", "Akamai",
    "Docker", "Kubernetes", "K8s", "Helm", "Istio", "Linkerd",
    "Podman", "Containerd", "Docker Compose",
    "GitHub Actions", "GitLab CI", "GitLab CI/CD", "Jenkins",
    "CircleCI", "Travis CI", "Bamboo", "TeamCity",
    "ArgoCD", "Argo CD", "FluxCD", "Spinnaker",
    "Terraform", "Pulumi", "CloudFormation", "CDK", "AWS CDK",
    "Ansible", "Chef", "Puppet", "SaltStack",
    "Linux", "Ubuntu", "CentOS", "Debian", "RHEL", "Amazon Linux",
    "Windows Server", "macOS",
    "S3", "EC2", "Lambda", "ECS", "EKS", "Fargate",
    "CloudFront", "Route 53", "API Gateway", "SQS", "SNS",
    "Step Functions", "EventBridge", "Cognito", "IAM",
    "VPC", "ELB", "ALB", "Cloud Functions", "Cloud Run",
    "App Engine", "Compute Engine", "GKE", "Cloud Storage",
    "Azure Functions", "Azure DevOps", "AKS", "Blob Storage",
    "Prometheus", "Grafana", "Datadog", "New Relic", "Splunk",
    "ELK Stack", "Logstash", "Kibana", "Fluentd", "Loki",
    "PagerDuty", "OpsGenie", "Sentry", "Honeycomb",
    "Vault", "HashiCorp Vault", "Consul", "Nomad",
    "Nginx", "HAProxy", "Traefik", "Envoy",
    "Cloudflare Workers", "Edge Functions", "CDN",
    "CI/CD", "Infrastructure as Code", "IaC",
    "Site Reliability Engineering", "SRE", "DevOps",
    "Platform Engineering", "Observability", "Monitoring",
    "Load Balancing", "Auto Scaling", "Microservices",
    "Service Mesh", "Serverless", "Container Orchestration",
]

_TOOLS = [
    "Git", "GitHub", "GitLab", "Bitbucket", "SVN",
    "VS Code", "Visual Studio Code", "Visual Studio",
    "IntelliJ IDEA", "IntelliJ", "PyCharm", "WebStorm",
    "Eclipse", "Xcode", "Android Studio", "Vim", "Neovim",
    "Emacs", "Sublime Text", "Atom",
    "Jupyter", "Jupyter Notebook", "JupyterLab", "Google Colab", "Colab",
    "Postman", "Insomnia", "cURL", "HTTPie",
    "Jira", "Confluence", "Trello", "Asana", "Linear", "Notion",
    "Monday.com", "ClickUp", "Basecamp",
    "Figma", "Sketch", "Adobe XD", "InVision", "Zeplin",
    "Canva", "Photoshop", "Illustrator",
    "Tableau", "Power BI", "PowerBI", "Looker", "Metabase",
    "Grafana", "Superset", "Apache Superset",
    "Streamlit", "Gradio", "Dash", "Panel",
    "Slack", "Discord", "Microsoft Teams", "Zoom",
    "npm", "yarn", "pnpm", "pip", "conda", "poetry", "pipenv",
    "Maven", "Gradle", "CMake", "Make", "Bazel",
    "Homebrew", "apt", "dnf", "snap", "Chocolatey",
    "Selenium", "Cypress", "Playwright", "Puppeteer",
    "Jest", "Mocha", "Chai", "Vitest", "Testing Library",
    "React Testing Library", "Enzyme",
    "pytest", "unittest", "nose", "tox", "coverage",
    "JUnit", "TestNG", "Mockito", "RSpec", "Minitest",
    "Postman", "SoapUI", "k6", "Locust", "JMeter", "Gatling",
    "Swagger UI", "Redoc",
    "Wireshark", "Fiddler", "Charles Proxy",
    "Chrome DevTools", "Lighthouse", "PageSpeed",
    "LaTeX", "Markdown", "Mermaid", "PlantUML",
    "RStudio", "SPSS", "SAS", "Stata",
    "Unity", "Unreal Engine", "Godot", "Blender",
    "AutoCAD", "SolidWorks", "CATIA",
    "QGIS", "ArcGIS",
    "Airflow", "Apache Airflow", "Prefect", "Dagster", "Luigi",
    "dbt", "Great Expectations",
    "Snowpark", "Apache Spark", "Spark", "Hadoop", "MapReduce",
    "Apache Flink", "Apache Beam", "Apache NiFi",
    "Twilio", "SendGrid", "Stripe", "PayPal",
    "Auth0", "Okta", "Keycloak", "Firebase Auth",
    "OAuth", "OAuth2", "JWT", "SAML", "OpenID Connect", "OIDC",
    "Agile", "Scrum", "Kanban", "SAFe",
    "TDD", "BDD", "DDD", "SOLID", "Design Patterns",
    "Microservices Architecture", "Event-Driven Architecture",
    "CQRS", "Event Sourcing", "Clean Architecture",
    "System Design", "API Design", "Database Design",
]

_SECURITY = [
    "Cybersecurity", "Information Security", "InfoSec",
    "Penetration Testing", "Ethical Hacking", "OWASP",
    "Burp Suite", "Metasploit", "Nmap", "Nessus",
    "Wireshark", "Kali Linux", "SOC", "SIEM",
    "Encryption", "TLS", "SSL", "PKI", "AES", "RSA",
    "Zero Trust", "Firewall", "IDS", "IPS",
    "Vulnerability Assessment", "Security Audit",
    "Compliance", "GDPR", "HIPAA", "SOC 2", "PCI DSS",
    "ISO 27001", "NIST", "CIS",
]

def _build_taxonomy() -> dict:
    """Build lowercase->category lookup dict from the lists above."""
    taxonomy = {}
    for skill in _PROGRAMMING_LANGUAGES:
        taxonomy[skill.lower()] = "Programming Languages"
    for skill in _AI_ML:
        taxonomy[skill.lower()] = "AI/ML"
    for skill in _WEB_FRAMEWORKS:
        taxonomy[skill.lower()] = "Web Frameworks"
    for skill in _DATABASES:
        taxonomy[skill.lower()] = "Databases"
    for skill in _CLOUD_DEVOPS:
        taxonomy[skill.lower()] = "Cloud/DevOps"
    for skill in _TOOLS:
        taxonomy[skill.lower()] = "Tools"
    for skill in _SECURITY:
        taxonomy[skill.lower()] = "Security"
    return taxonomy

TECH_SKILLS = _build_taxonomy()

# Canonical casing: lowercase -> preferred display form
_CANONICAL_CASE = {}
for _lst in [_PROGRAMMING_LANGUAGES, _AI_ML, _WEB_FRAMEWORKS, _DATABASES,
             _CLOUD_DEVOPS, _TOOLS, _SECURITY]:
    for _s in _lst:
        _CANONICAL_CASE[_s.lower()] = _s

# Words that signal a skill entry is actually a category header
_HEADER_SIGNALS = re.compile(
    r'\b(stack|engineering|tools|frameworks|languages|development|technologies|'
    r'platforms|libraries|infrastructure|services|proficiencies|competencies|'
    r'expertise|skillset|skill set|database|devops|frontend|backend|'
    r'front-end|back-end|full-stack|fullstack|cloud|data|web|mobile|'
    r'software|programming|testing|automation|management|design|analytics|'
    r'machine learning|artificial intelligence|ml/ai|ai/ml)\b',
    re.IGNORECASE
)


# ── Main entry point ──────────────────────────────────────────────────────

def post_process_resume(raw_text: str, parsed: dict) -> dict:
    """
    Run deterministic post-processing stages on an LLM-parsed resume.

    Args:
        raw_text: The original plain-text extracted from the resume file.
        parsed:   The dict produced by json.loads() of the LLM output.

    Returns:
        The parsed dict, mutated in-place with fixes applied.
    """
    skills = parsed.get("skills") or []

    # Stage 1 — Expand category-header skills into individual skills
    skills = _fix_category_header_skills(raw_text, skills)

    # Stage 2 — Mine raw text for known tech skills the LLM missed
    skills = _extract_skills_from_text(raw_text, skills)

    parsed["skills"] = skills

    # Build a quick set of all known skill names for experience scanning
    all_skill_names = {s["skill_name"].lower() for s in skills}
    all_skill_names.update(TECH_SKILLS.keys())

    # Stage 3 — Populate empty skills_used in experience entries
    experience = parsed.get("experience") or []
    _extract_skills_used_from_experience(experience, all_skill_names)
    parsed["experience"] = experience

    # Stage 4 — Generate summary if missing
    _generate_summary_if_missing(parsed)

    logger.info(
        "Post-processed resume: %d skills, %d experience entries patched",
        len(parsed["skills"]),
        sum(1 for e in experience if e.get("skills_used")),
    )
    return parsed


# ── Stage 1: Fix category-header skills ───────────────────────────────────

def _fix_category_header_skills(raw_text: str, skills: list) -> list:
    """
    Detect skill entries that are actually category headers and expand them
    into individual skills by finding the matching line in raw_text.
    """
    expanded = []
    raw_lower = raw_text.lower()

    for skill_entry in skills:
        name = skill_entry.get("skill_name", "")
        # Check if this looks like a category header
        if not _is_category_header(name):
            expanded.append(skill_entry)
            continue

        # Try to find this header in the raw text and extract individual skills
        individual = _expand_header_from_text(raw_text, raw_lower, name)
        if individual:
            for ind_name in individual:
                expanded.append({
                    "skill_name": ind_name,
                    "proficiency": skill_entry.get("proficiency"),
                    "category": name,  # use the header as category
                    "years_of_experience": skill_entry.get("years_of_experience"),
                })
        else:
            # Couldn't expand — keep original so we don't lose data
            expanded.append(skill_entry)

    return expanded


def _is_category_header(name: str) -> bool:
    """Return True if the skill name looks like a category header."""
    if not name:
        return False
    # If it contains a colon, it's very likely a header ("AI & LLM Stack:")
    if ":" in name:
        return True
    # If it matches known header signal words AND has > 1 word
    words = name.split()
    if len(words) >= 2 and _HEADER_SIGNALS.search(name):
        return True
    # If the skill name has "&" plus signal words
    if "&" in name and _HEADER_SIGNALS.search(name):
        return True
    return False


def _expand_header_from_text(raw_text: str, raw_lower: str, header: str) -> list:
    """
    Find the header line in raw_text and extract the comma-separated skills
    that follow it (on the same line or after a colon).
    """
    # Clean header for search (remove trailing colon if present)
    search_header = header.rstrip(":").strip().lower()

    # Try to find a line like "Header: skill1, skill2, skill3"
    # or "Header — skill1, skill2, skill3"
    for line in raw_text.splitlines():
        line_lower = line.lower().strip()
        if search_header not in line_lower:
            continue

        # Extract the part after the header
        # Try colon, em-dash, en-dash, pipe as separators
        after = None
        for sep in [":", "–", "—", "-", "|"]:
            idx = line.find(sep)
            if idx >= 0:
                header_part = line[:idx].lower().strip()
                # Make sure the header part actually contains our search term
                if search_header in header_part or header_part in search_header:
                    after = line[idx + len(sep):]
                    break

        if not after:
            continue

        # Split by commas, semicolons, pipes, or " | "
        items = re.split(r'[,;|]', after)
        result = []
        for item in items:
            item = item.strip().strip("•·-–—")
            if not item:
                continue
            # Handle parenthetical sub-items: "Vector DBs (FAISS, Chroma)"
            paren_match = re.match(r'^(.+?)\s*\((.+)\)\s*$', item)
            if paren_match:
                main = paren_match.group(1).strip()
                subs = paren_match.group(2).split(",")
                if main and len(main) > 1:
                    result.append(main)
                for sub in subs:
                    sub = sub.strip()
                    if sub and len(sub) > 1:
                        result.append(sub)
            else:
                if len(item) > 1:
                    result.append(item)

        if result:
            return result

    return []


# ── Stage 2: Extract skills from raw text ─────────────────────────────────

def _extract_skills_from_text(raw_text: str, existing_skills: list) -> list:
    """
    Scan raw_text for known tech skills not already in the parsed skills list.
    """
    existing_lower = {s.get("skill_name", "").lower() for s in existing_skills}
    raw_lower = raw_text.lower()

    # For multi-word skills, use simple substring matching.
    # For single-word skills, use word-boundary matching to avoid false positives.
    added = set()
    for skill_lower, category in TECH_SKILLS.items():
        if skill_lower in existing_lower:
            continue
        if skill_lower in added:
            continue

        words = skill_lower.split()
        if len(words) == 1:
            # Word-boundary match for single-word skills
            pattern = r'\b' + re.escape(skill_lower) + r'\b'
            if re.search(pattern, raw_lower):
                display = _CANONICAL_CASE.get(skill_lower, skill_lower)
                existing_skills.append({
                    "skill_name": display,
                    "proficiency": None,
                    "category": category,
                    "years_of_experience": None,
                })
                added.add(skill_lower)
        else:
            # Multi-word: just check substring presence
            if skill_lower in raw_lower:
                display = _CANONICAL_CASE.get(skill_lower, skill_lower)
                existing_skills.append({
                    "skill_name": display,
                    "proficiency": None,
                    "category": category,
                    "years_of_experience": None,
                })
                added.add(skill_lower)

    return existing_skills


# ── Stage 3: Extract skills_used from experience ──────────────────────────

def _extract_skills_used_from_experience(experience: list, all_skills: set):
    """
    For each experience entry with empty skills_used, scan responsibilities
    for known skills and populate the field.
    """
    for entry in experience:
        existing_used = entry.get("skills_used") or []
        if existing_used:
            continue

        responsibilities = entry.get("responsibilities") or []
        text = " ".join(responsibilities).lower()
        if not text:
            continue

        found = []
        for skill_lower in all_skills:
            words = skill_lower.split()
            if len(words) == 1:
                pattern = r'\b' + re.escape(skill_lower) + r'\b'
                if re.search(pattern, text):
                    display = _CANONICAL_CASE.get(skill_lower, skill_lower)
                    found.append(display)
            else:
                if skill_lower in text:
                    display = _CANONICAL_CASE.get(skill_lower, skill_lower)
                    found.append(display)

        if found:
            entry["skills_used"] = sorted(set(found))


# ── Stage 4: Generate summary if missing ──────────────────────────────────

def _generate_summary_if_missing(parsed: dict):
    """Build a summary from available fields if none exists."""
    summary = (parsed.get("summary") or "").strip()
    if summary:
        return

    parts = []

    # Title
    basic = parsed.get("basic_info") or {}
    title = basic.get("current_title")
    if title:
        parts.append(title)

    # Years of experience
    experience = parsed.get("experience") or []
    total_years = _estimate_total_years(experience)
    if total_years > 0:
        parts.append(f"with {total_years}+ years of experience")

    # Top skills
    skills = parsed.get("skills") or []
    top_skills = [s.get("skill_name", "") for s in skills[:8] if s.get("skill_name")]
    if top_skills:
        parts.append("skilled in " + ", ".join(top_skills))

    if parts:
        parsed["summary"] = " ".join(parts) + "."


def _estimate_total_years(experience: list) -> int:
    """Estimate total years of experience from the experience entries."""
    if not experience:
        return 0

    current_year = datetime.now().year
    earliest = current_year
    latest = 0

    for entry in experience:
        start = entry.get("start_date") or ""
        end = entry.get("end_date") or ""

        start_year = _extract_year(start)
        end_year = _extract_year(end)
        if not end_year and end.lower() in ("present", "current", "now", ""):
            end_year = current_year

        if start_year:
            earliest = min(earliest, start_year)
        if end_year:
            latest = max(latest, end_year)

    if latest > earliest:
        return latest - earliest
    return 0


def _extract_year(date_str: str) -> int:
    """Extract a 4-digit year from a date string."""
    if not date_str:
        return 0
    match = re.search(r'(19|20)\d{2}', date_str)
    return int(match.group()) if match else 0
