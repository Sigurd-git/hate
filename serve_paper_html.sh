#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_TEX="${PROJECT_ROOT}/slides/paper.tex"
BUILD_DIR="${PROJECT_ROOT}/slides/html"
TMP_TEX="${BUILD_DIR}/paper.html-source.tex"
OUTPUT_HTML="${BUILD_DIR}/paper.html"
BIND_HOST="${BIND_HOST:-10.40.61.207}"
PORT="${1:-${PORT:-8088}}"

if [[ ! "${PORT}" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [port]" >&2
  echo "Example: $0 8088" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

echo "[1/5] Generating paper figures"
if [[ -f "${PROJECT_ROOT}/make_paper_revision_figures.py" ]]; then
  uv run python make_paper_revision_figures.py
else
  echo "  make_paper_revision_figures.py not found; skipping figure regeneration"
fi

echo "[2/5] Preparing HTML build directory"
mkdir -p "${BUILD_DIR}"

cat > "${BUILD_DIR}/paper.css" <<'CSS'
:root {
  color-scheme: light;
  --text: #1f2933;
  --muted: #5b6776;
  --rule: #d8dee8;
  --link: #1d4f91;
  --background: #ffffff;
}

html {
  background: #f5f7fa;
}

body {
  box-sizing: border-box;
  max-width: 980px;
  margin: 0 auto;
  padding: 3rem 1.5rem 4rem;
  background: var(--background);
  color: var(--text);
  font-family: "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  font-size: 17px;
  line-height: 1.72;
}

h1, h2, h3, h4 {
  line-height: 1.28;
  color: #111827;
}

h1 {
  font-size: 2rem;
  margin-bottom: 1.4rem;
}

h2 {
  margin-top: 2.6rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--rule);
}

h3 {
  margin-top: 2rem;
}

a {
  color: var(--link);
}

figure {
  margin: 2rem 0;
}

img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
}

figcaption {
  margin-top: 0.8rem;
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.55;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.8rem 0;
  font-size: 0.92rem;
}

th, td {
  border-bottom: 1px solid var(--rule);
  padding: 0.45rem 0.55rem;
  vertical-align: top;
}

th {
  text-align: left;
  color: #111827;
}

blockquote {
  margin: 1.5rem 0;
  padding-left: 1rem;
  border-left: 4px solid var(--rule);
  color: var(--muted);
}

code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.94em;
}

.math {
  overflow-x: auto;
}

@media (max-width: 720px) {
  body {
    padding: 1.5rem 1rem 3rem;
    font-size: 16px;
  }

  h1 {
    font-size: 1.55rem;
  }

  table {
    display: block;
    overflow-x: auto;
    white-space: nowrap;
  }
}
CSS

echo "[3/5] Converting PDF figure references to PNG for browser display"
sed 's/\.pdf}/.png}/g' "${PAPER_TEX}" > "${TMP_TEX}"
cp -f "${PROJECT_ROOT}"/artifacts/paper_revision/fig_r*.png "${BUILD_DIR}/"

echo "[4/5] Compiling LaTeX to standalone HTML"
pandoc "${TMP_TEX}" \
  --from=latex \
  --to=html5 \
  --standalone \
  --mathml \
  --resource-path="${PROJECT_ROOT}:${PROJECT_ROOT}/artifacts/paper_revision:${PROJECT_ROOT}/artifacts/paper_followups:${PROJECT_ROOT}/artifacts/human_model_dz_descriptive_comparison:${PROJECT_ROOT}/human/outputs/final_clean_long_gender_difference_analysis/figures" \
  --css=paper.css \
  --metadata title="Measurement-Conditional Alignment of Human and LLM Gender Bias in Attack Ratings" \
  --output="${OUTPUT_HTML}"

echo "[5/5] Serving ${BUILD_DIR}"
echo "URL: http://${BIND_HOST}:${PORT}/paper.html"
cd "${BUILD_DIR}"
exec uv run python -m http.server "${PORT}" --bind "${BIND_HOST}"
