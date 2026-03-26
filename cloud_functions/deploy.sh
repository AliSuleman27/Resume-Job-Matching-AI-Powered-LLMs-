#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CF_DIR="$PROJECT_ROOT/cloud_functions"
AI_DIR="$CF_DIR/ai_matching"

# ── Copy service files into ai_matching/ ─────────────────────────────
echo "==> Copying services, models, and data into ai_matching/..."

# Services
mkdir -p "$AI_DIR/services"
cp "$PROJECT_ROOT/services/__init__.py"              "$AI_DIR/services/"
cp "$PROJECT_ROOT/services/hybrid_matcher.py"        "$AI_DIR/services/"
cp "$PROJECT_ROOT/services/constraint_matcher.py"    "$AI_DIR/services/"
cp "$PROJECT_ROOT/services/embedding_service.py"     "$AI_DIR/services/"
cp "$PROJECT_ROOT/services/skill_graph.py"           "$AI_DIR/services/"
cp "$PROJECT_ROOT/services/resume_post_processor.py" "$AI_DIR/services/"
cp "$PROJECT_ROOT/services/mongo_service.py"         "$AI_DIR/services/"
cp "$PROJECT_ROOT/services/questionnaire_scorer.py"  "$AI_DIR/services/"
cp "$PROJECT_ROOT/services/type_convertor.py"        "$AI_DIR/services/"

# Models
mkdir -p "$AI_DIR/models"
cp "$PROJECT_ROOT/models/__init__.py"                "$AI_DIR/models/"
cp "$PROJECT_ROOT/models/resume_model.py"            "$AI_DIR/models/"
cp "$PROJECT_ROOT/models/job_description_model.py"   "$AI_DIR/models/"

# Data (ontology files)
mkdir -p "$AI_DIR/data"
cp "$PROJECT_ROOT/data/mind_skills.json"             "$AI_DIR/data/"
cp "$PROJECT_ROOT/data/esco_digital_skills.csv"      "$AI_DIR/data/"

echo "==> Files copied."

# ── Deploy AI Matching Cloud Function ────────────────────────────────
echo "==> Deploying cf-ai-matching..."
gcloud functions deploy cf-ai-matching \
    --gen2 \
    --runtime=python312 \
    --region=us-central1 \
    --source="$AI_DIR" \
    --entry-point=match_candidates \
    --trigger-http \
    --allow-unauthenticated \
    --timeout=540s \
    --memory=1Gi \
    --max-instances=3 \
    --set-env-vars="MONGO_URI=${MONGO_URI},HF_TOKEN=${HF_TOKEN},CF_AUTH_TOKEN=${CF_AUTH_TOKEN}"

echo "==> cf-ai-matching deployed."

# ── Deploy Email Notifications Cloud Function ────────────────────────
echo "==> Deploying cf-email-notifications..."
gcloud functions deploy cf-email-notifications \
    --gen2 \
    --runtime=python312 \
    --region=us-central1 \
    --source="$CF_DIR/email_notifications" \
    --entry-point=send_notification \
    --trigger-http \
    --allow-unauthenticated \
    --timeout=60s \
    --memory=256Mi \
    --max-instances=10 \
    --set-env-vars="MONGO_URI=${MONGO_URI},MAIL_USERNAME=${MAIL_USERNAME},MAIL_PASSWORD=${MAIL_PASSWORD},CF_AUTH_TOKEN=${CF_AUTH_TOKEN}"

echo "==> cf-email-notifications deployed."
echo ""
echo "Done! Both Cloud Functions are deployed."
echo "AI Matching URL:  $(gcloud functions describe cf-ai-matching --gen2 --region=us-central1 --format='value(serviceConfig.uri)' 2>/dev/null || echo 'run: gcloud functions describe cf-ai-matching --gen2 --region=us-central1')"
echo "Email URL:        $(gcloud functions describe cf-email-notifications --gen2 --region=us-central1 --format='value(serviceConfig.uri)' 2>/dev/null || echo 'run: gcloud functions describe cf-email-notifications --gen2 --region=us-central1')"
