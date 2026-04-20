#!/usr/bin/env bash
# =============================================================================
# FOUNDRY IGNITION PROTOCOL — Neral AI v6.1
# Principal Security Architect Directive
#
# PURPOSE : Initialize, audit, and ignite the local development environment
#           for the Neral AI Spotlight stack with zero credential leakage.
#
# EXECUTION: chmod +x foundry_setup.sh && ./foundry_setup.sh
#
# STAGES  : [1] INITIALIZE  — .env scaffold and key validation
#           [2] AUDIT       — .gitignore enforcement
#           [3] IGNITE      — single-command frontend launch instruction
#
# PLATFORM: Ubuntu (bash >= 4.0)
# =============================================================================

set -euo pipefail   # Abort on error, undefined var, or pipe failure

# ── TERMINAL COLOR CODES ─────────────────────────────────────────────────────
RED='\033[0;31m'
YLW='\033[1;33m'
GRN='\033[0;32m'
CYN='\033[0;36m'
BLD='\033[1m'
RST='\033[0m'

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
GITIGNORE_FILE="${REPO_ROOT}/.gitignore"
FRONTEND_ENTRY="${REPO_ROOT}/spotlight/frontend_skeleton.py"
REQUIRED_KEY="NERAL_API_KEY"
LEGACY_KEY="API_KEY"                  # Old key name — triggers a warning
PLACEHOLDER="REPLACE_WITH_REAL_KEY"  # Sentinel; ignition will block on this

echo ""
echo -e "${BLD}${CYN}╔══════════════════════════════════════════════════════════╗${RST}"
echo -e "${BLD}${CYN}║        NERAL AI · FOUNDRY IGNITION PROTOCOL v6.1        ║${RST}"
echo -e "${BLD}${CYN}╚══════════════════════════════════════════════════════════╝${RST}"
echo ""


# =============================================================================
# STAGE 1 — INITIALIZE
# Verify .env exists and contains the correct key (NERAL_API_KEY).
# If absent, scaffold it. If using legacy key name, issue a hard warning.
# =============================================================================
echo -e "${BLD}[ STAGE 1 ]${RST} INITIALIZE — .env integrity check"
echo "──────────────────────────────────────────────────────────"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo -e "  ${YLW}▶ .env not found. Scaffolding from template...${RST}"
    cat > "${ENV_FILE}" <<EOF
# =============================================================================
# Neral AI — Local Environment Configuration
# SECURITY: This file must NEVER be committed to version control.
#           Verify .gitignore coverage before any 'git add' operation.
# =============================================================================

# Primary authentication key for the Secured REST Gateway.
# Replace the placeholder below with your actual secret before running.
${REQUIRED_KEY}=${PLACEHOLDER}

# Model artifact paths (read by app/main.py lifespan loader)
MODEL_PATH_ECOMM=models/hcim_E-comm_Stream_v1.joblib
MODEL_PATH_AVIATION=models/hcim_Aviation_v1.joblib
EOF
    echo -e "  ${GRN}✔ .env created at: ${ENV_FILE}${RST}"
    echo -e "  ${RED}▲ ACTION REQUIRED: Open .env and replace '${PLACEHOLDER}' with your actual key.${RST}"
else
    echo -e "  ${GRN}✔ .env exists at: ${ENV_FILE}${RST}"

    # Check for legacy key name
    if grep -q "^${LEGACY_KEY}=" "${ENV_FILE}" && ! grep -q "^${REQUIRED_KEY}=" "${ENV_FILE}"; then
        echo ""
        echo -e "  ${RED}╔══ SECURITY WARNING ═══════════════════════════════════════╗${RST}"
        echo -e "  ${RED}║  Legacy key '${LEGACY_KEY}' detected.                            ║${RST}"
        echo -e "  ${RED}║  The application requires '${REQUIRED_KEY}'.                  ║${RST}"
        echo -e "  ${RED}║                                                           ║${RST}"
        echo -e "  ${RED}║  ACTION: Rename the key in .env:                          ║${RST}"
        echo -e "  ${RED}║    OLD → ${LEGACY_KEY}=<value>                                 ║${RST}"
        echo -e "  ${RED}║    NEW → ${REQUIRED_KEY}=<value>                             ║${RST}"
        echo -e "  ${RED}╚═══════════════════════════════════════════════════════════╝${RST}"
        echo ""
    fi

    # Check for unfilled placeholder
    if grep -q "${PLACEHOLDER}" "${ENV_FILE}"; then
        echo -e "  ${RED}▲ WARNING: .env contains the placeholder sentinel '${PLACEHOLDER}'.${RST}"
        echo -e "  ${RED}  Replace it with a real key before proceeding to STAGE 3.${RST}"
    else
        # Validate the key actually has a non-empty value
        KEY_VALUE=$(grep "^${REQUIRED_KEY}=" "${ENV_FILE}" | cut -d'=' -f2- | tr -d '[:space:]')
        if [[ -z "${KEY_VALUE}" ]]; then
            echo -e "  ${RED}▲ CRITICAL: ${REQUIRED_KEY} is present but empty in .env.${RST}"
        else
            echo -e "  ${GRN}✔ ${REQUIRED_KEY} is set and non-empty.${RST}"
        fi
    fi
fi

echo ""


# =============================================================================
# STAGE 2 — AUDIT
# Confirm .env is covered by .gitignore.
# Uses two checks:
#   (a) grep for a literal ".env" pattern line
#   (b) git check-ignore (requires repo context) as a secondary oracle
# Appends a block only if genuinely absent — never blindly appends.
# =============================================================================
echo -e "${BLD}[ STAGE 2 ]${RST} AUDIT — .gitignore enforcement"
echo "──────────────────────────────────────────────────────────"

if [[ ! -f "${GITIGNORE_FILE}" ]]; then
    echo -e "  ${YLW}▶ .gitignore not found. Creating and appending .env rule.${RST}"
    cat > "${GITIGNORE_FILE}" <<EOF
# Created by Foundry Ignition Protocol
.env
.envrc
EOF
    echo -e "  ${GRN}✔ .gitignore scaffolded with .env rule.${RST}"
else
    # Primary check: literal pattern match (handles both '/.env' and '.env')
    if grep -qE '^\/?\.env$' "${GITIGNORE_FILE}"; then
        echo -e "  ${GRN}✔ .env is indexed in .gitignore — no action required.${RST}"
    else
        # Secondary check: ask git directly
        if git -C "${REPO_ROOT}" check-ignore -q .env 2>/dev/null; then
            echo -e "  ${GRN}✔ .env is suppressed by git (wildcard or inherited rule).${RST}"
        else
            echo -e "  ${YLW}▶ .env is NOT covered by .gitignore. Appending rule now...${RST}"
            printf '\n# Neral AI — Foundry Ignition: credential file lockout\n.env\n.envrc\n' >> "${GITIGNORE_FILE}"
            echo -e "  ${GRN}✔ .env rule appended to .gitignore.${RST}"
        fi
    fi
fi

# Tertiary guard: verify .env is not currently staged
if git -C "${REPO_ROOT}" ls-files --error-unmatch .env &>/dev/null; then
    echo ""
    echo -e "  ${RED}╔══ STAGING BREACH DETECTED ════════════════════════════════╗${RST}"
    echo -e "  ${RED}║  .env is tracked by git. This is a credential leak vector.║${RST}"
    echo -e "  ${RED}║                                                           ║${RST}"
    echo -e "  ${RED}║  REMEDIATION — run immediately:                           ║${RST}"
    echo -e "  ${RED}║    git rm --cached .env                                   ║${RST}"
    echo -e "  ${RED}║    git commit -m 'security: untrack .env from index'      ║${RST}"
    echo -e "  ${RED}╚═══════════════════════════════════════════════════════════╝${RST}"
else
    echo -e "  ${GRN}✔ .env is not staged in git index — clean.${RST}"
fi

echo ""


# =============================================================================
# STAGE 3 — IGNITE
# Block launch if placeholder is still present.
# Otherwise emit the single-command ignition string.
# =============================================================================
echo -e "${BLD}[ STAGE 3 ]${RST} IGNITE — launch command"
echo "──────────────────────────────────────────────────────────"

# Re-read key value after all prior mutations
LIVE_KEY_VALUE=$(grep "^${REQUIRED_KEY}=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2- | tr -d '[:space:]' || true)

if [[ -z "${LIVE_KEY_VALUE}" ]] || [[ "${LIVE_KEY_VALUE}" == "${PLACEHOLDER}" ]]; then
    echo ""
    echo -e "  ${RED}╔══ IGNITION BLOCKED ════════════════════════════════════════╗${RST}"
    echo -e "  ${RED}║  ${REQUIRED_KEY} is not populated in .env.               ║${RST}"
    echo -e "  ${RED}║  The Secured REST Gateway will raise EnvironmentError     ║${RST}"
    echo -e "  ${RED}║  on startup. Ignition refused.                            ║${RST}"
    echo -e "  ${RED}║                                                           ║${RST}"
    echo -e "  ${RED}║  FIX: Edit .env → set ${REQUIRED_KEY}=<your_key>          ║${RST}"
    echo -e "  ${RED}║  Then re-run: ./foundry_setup.sh                          ║${RST}"
    echo -e "  ${RED}╚═══════════════════════════════════════════════════════════╝${RST}"
    echo ""
    exit 1
fi

echo -e "  ${GRN}✔ Key validated. Emitting ignition command string.${RST}"
echo ""
echo -e "${BLD}${CYN}  ── SINGLE-COMMAND IGNITION ────────────────────────────────${RST}"
echo ""
echo -e "  ${BLD}export \$(grep -v '^#' .env | xargs) && streamlit run ${FRONTEND_ENTRY} \\"
echo -e "    --server.port 8501 --server.address 0.0.0.0 --server.headless true${RST}"
echo ""
echo -e "${CYN}  ── OR source-based (persists for shell session) ───────────${RST}"
echo ""
echo -e "  ${BLD}set -a && source .env && set +a && streamlit run ${FRONTEND_ENTRY} \\"
echo -e "    --server.port 8501 --server.address 0.0.0.0 --server.headless true${RST}"
echo ""
echo -e "  ${YLW}Note: Port 8501 is the local dev binding.${RST}"
echo -e "  ${YLW}      Ensure the FastAPI backend is running on port 8000 before executing.${RST}"
echo -e "  ${YLW}      FastAPI launch: uvicorn app.main:app --host 0.0.0.0 --port 8000${RST}"
echo ""

echo -e "${BLD}${GRN}╔══════════════════════════════════════════════════════════╗${RST}"
echo -e "${BLD}${GRN}║          FOUNDRY IGNITION PROTOCOL — COMPLETE            ║${RST}"
echo -e "${BLD}${GRN}╚══════════════════════════════════════════════════════════╝${RST}"
echo ""
