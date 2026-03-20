#!/bin/bash
BASE="http://localhost:5000"
COOKIE_DIR="/tmp/test_cookies"
DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$COOKIE_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[OK]${NC} $1"; }
err() { echo -e "${RED}[FAIL]${NC} $1"; }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# =============================================
# 1. CREATE RECRUITER ACCOUNT
# =============================================
info "=== Step 1: Signup Recruiter ==="
RESP=$(curl -s -c "$COOKIE_DIR/recruiter.txt" -b "$COOKIE_DIR/recruiter.txt" \
  -X POST "$BASE/signup_recruiter" \
  -d "email=alisuleman@gmail.com&password=12345678&confirm_password=12345678&name=Ali Suleman&company=TechVentures Pakistan" \
  -w "\n%{http_code}" -L)
CODE=$(echo "$RESP" | tail -1)
info "Recruiter signup returned $CODE (OK if already exists)"

# Login as recruiter
info "=== Logging in as Recruiter ==="
curl -s -c "$COOKIE_DIR/recruiter.txt" -b "$COOKIE_DIR/recruiter.txt" \
  -X POST "$BASE/login_recruiter" \
  -d "email=alisuleman@gmail.com&password=12345678" -L -o /dev/null
log "Recruiter logged in"

# =============================================
# 2. CREATE JOB DESCRIPTIONS
# =============================================
info "=== Step 2: Upload Job Descriptions ==="

info "Uploading: Senior Python Developer JD..."
RESP=$(curl -s -c "$COOKIE_DIR/recruiter.txt" -b "$COOKIE_DIR/recruiter.txt" \
  -X POST "$BASE/recruiter/create_job" \
  -F "file=@${DATA_DIR}/job_senior_python.txt" \
  -L -w "\n%{http_code}")
CODE=$(echo "$RESP" | tail -1)
if [[ "$CODE" == "200" ]]; then log "Job 1 (Senior Python Dev) uploaded"; else err "Job 1 upload returned $CODE"; fi

info "Uploading: ML Engineer JD..."
RESP=$(curl -s -c "$COOKIE_DIR/recruiter.txt" -b "$COOKIE_DIR/recruiter.txt" \
  -X POST "$BASE/recruiter/create_job" \
  -F "file=@${DATA_DIR}/job_ml_engineer.txt" \
  -L -w "\n%{http_code}")
CODE=$(echo "$RESP" | tail -1)
if [[ "$CODE" == "200" ]]; then log "Job 2 (ML Engineer) uploaded"; else err "Job 2 upload returned $CODE"; fi

# Get job IDs from dashboard HTML
info "=== Fetching Job IDs ==="
DASHBOARD=$(curl -s -c "$COOKIE_DIR/recruiter.txt" -b "$COOKIE_DIR/recruiter.txt" "$BASE/recruiter/dashboard")
JOB_IDS=$(echo "$DASHBOARD" | grep -oE '/job/[a-f0-9]{24}' | grep -oE '[a-f0-9]{24}' | sort -u)
JOB_ARRAY=($JOB_IDS)
info "Found ${#JOB_ARRAY[@]} jobs: ${JOB_ARRAY[*]}"

# Logout recruiter
curl -s -b "$COOKIE_DIR/recruiter.txt" "$BASE/logout" -o /dev/null

# =============================================
# 3. CREATE CANDIDATE ACCOUNTS & UPLOAD RESUMES
# =============================================
declare -A CANDIDATES
CANDIDATES=(
  ["ahmed"]="ahmed.raza@email.com"
  ["fatima"]="fatima.malik@email.com"
  ["bilal"]="bilal.hassan@email.com"
  ["ayesha"]="ayesha.siddiqui@email.com"
  ["usman"]="usman.tariq@email.com"
  ["zainab"]="zainab.qureshi@email.com"
)

declare -A RESUME_IDS

for name in ahmed fatima bilal ayesha usman zainab; do
  email="${CANDIDATES[$name]}"
  info "=== Candidate: $name ($email) ==="

  # Signup
  curl -s -c "$COOKIE_DIR/${name}.txt" -b "$COOKIE_DIR/${name}.txt" \
    -X POST "$BASE/signup" \
    -d "email=${email}&password=test1234" -L -o /dev/null
  log "$name signed up"

  # Login
  curl -s -c "$COOKIE_DIR/${name}.txt" -b "$COOKIE_DIR/${name}.txt" \
    -X POST "$BASE/login" \
    -d "email=${email}&password=test1234" -L -o /dev/null
  log "$name logged in"

  # Upload resume
  info "Uploading resume for $name..."
  RESP=$(curl -s -c "$COOKIE_DIR/${name}.txt" -b "$COOKIE_DIR/${name}.txt" \
    -X POST "$BASE/upload" \
    -F "file=@${DATA_DIR}/resume_${name}.txt" \
    -w "\n%{http_code}")
  CODE=$(echo "$RESP" | tail -1)
  BODY=$(echo "$RESP" | sed '$d')

  if [[ "$CODE" == "200" ]]; then
    log "$name resume uploaded successfully"
  else
    err "$name resume upload failed ($CODE)"
    echo "$BODY" | head -5
  fi

  # Get resume ID from dashboard HTML
  RESUME_ID=$(curl -s -c "$COOKIE_DIR/${name}.txt" -b "$COOKIE_DIR/${name}.txt" \
    "$BASE/dashboard" | grep -oE '/resume/[a-f0-9]{24}' | grep -oE '[a-f0-9]{24}' | head -1)
  RESUME_IDS[$name]="$RESUME_ID"
  if [[ -n "$RESUME_ID" ]]; then
    log "$name resume ID: $RESUME_ID"
  else
    err "Could not find resume ID for $name"
  fi
done

# =============================================
# 4. APPLY TO JOBS
# =============================================
info "=== Step 4: Applying to Jobs ==="

for name in ahmed fatima bilal ayesha usman zainab; do
  email="${CANDIDATES[$name]}"
  RESUME_ID="${RESUME_IDS[$name]}"
  if [[ -z "$RESUME_ID" ]]; then
    err "Skipping $name - no resume ID"
    continue
  fi

  # Re-login to be safe
  curl -s -c "$COOKIE_DIR/${name}.txt" -b "$COOKIE_DIR/${name}.txt" \
    -X POST "$BASE/login" \
    -d "email=${email}&password=test1234" -L -o /dev/null

  for JOB_ID in "${JOB_ARRAY[@]}"; do
    info "$name applying to job $JOB_ID..."
    RESP=$(curl -s -c "$COOKIE_DIR/${name}.txt" -b "$COOKIE_DIR/${name}.txt" \
      -X POST "$BASE/jobs/${JOB_ID}/apply" \
      -H "Content-Type: application/json" \
      -d "{\"resume_id\": \"${RESUME_ID}\", \"cover_message\": \"I am excited to apply for this position. My skills and experience align well with the requirements.\"}" \
      -w "\n%{http_code}")
    CODE=$(echo "$RESP" | tail -1)
    BODY=$(echo "$RESP" | sed '$d')
    if [[ "$CODE" == "200" ]]; then
      log "$name applied to job $JOB_ID"
    else
      err "$name apply failed ($CODE)"
    fi
  done

  # Logout
  curl -s -b "$COOKIE_DIR/${name}.txt" "$BASE/logout" -o /dev/null
done

# =============================================
# 5. RUN AI ENGINE
# =============================================
info "=== Step 5: Run AI Engine ==="

# Login as recruiter
curl -s -c "$COOKIE_DIR/recruiter.txt" -b "$COOKIE_DIR/recruiter.txt" \
  -X POST "$BASE/login_recruiter" \
  -d "email=alisuleman@gmail.com&password=12345678" -L -o /dev/null
log "Recruiter logged back in"

for JOB_ID in "${JOB_ARRAY[@]}"; do
  info "Running AI engine for job $JOB_ID (this may take a minute)..."
  RESP=$(curl -s -c "$COOKIE_DIR/recruiter.txt" -b "$COOKIE_DIR/recruiter.txt" \
    -X POST "$BASE/recruiter/job/${JOB_ID}/run_ai_engine" \
    -d "confirm_rerun=true" \
    -L -w "\n%{http_code}" --max-time 300)
  CODE=$(echo "$RESP" | tail -1)
  if [[ "$CODE" == "200" ]]; then
    log "AI engine completed for job $JOB_ID"
  else
    err "AI engine for job $JOB_ID returned $CODE"
  fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  TEST DATA CREATION COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Recruiter: alisuleman@gmail.com / 12345678"
echo "Candidates (password: test1234):"
echo "  - ahmed.raza@email.com"
echo "  - fatima.malik@email.com"
echo "  - bilal.hassan@email.com"
echo "  - ayesha.siddiqui@email.com"
echo "  - usman.tariq@email.com"
echo "  - zainab.qureshi@email.com"
echo "Jobs: ${#JOB_ARRAY[@]} created"
echo "AI Engine: Ran for all jobs"
