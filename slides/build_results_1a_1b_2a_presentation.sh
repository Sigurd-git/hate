#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tex_file="${script_dir}/results_1a_1b_2a_presentation.tex"
output_pdf="${script_dir}/results_1a_1b_2a_presentation.pdf"
manual_pdf="${script_dir}/../鲍璇1a1b2a结果汇报(1).pdf"
override_page_number=18

if [[ ! -f "${tex_file}" ]]; then
  echo "TeX file not found: ${tex_file}" >&2
  exit 1
fi

if [[ ! -f "${manual_pdf}" ]]; then
  echo "Manual PDF not found: ${manual_pdf}" >&2
  exit 1
fi

build_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${build_dir}"
}
trap cleanup EXIT

run_xelatex() {
  local pass_log="$1"
  local exit_code=0

  if ! xelatex -interaction=nonstopmode results_1a_1b_2a_presentation.tex >"${pass_log}" 2>&1; then
    exit_code=$?
  fi

  if grep -q "Output written on results_1a_1b_2a_presentation.pdf" "${pass_log}"; then
    return 0
  fi

  cat "${pass_log}" >&2
  echo "xelatex failed with exit code ${exit_code}" >&2
  exit 1
}

pushd "${script_dir}" >/dev/null
run_xelatex "${build_dir}/xelatex-pass1.log"
run_xelatex "${build_dir}/xelatex-pass2.log"
popd >/dev/null

page_count="$(pdfinfo "${output_pdf}" | awk '/^Pages:/ {print $2}')"
if [[ -z "${page_count}" ]]; then
  echo "Failed to read page count from ${output_pdf}" >&2
  exit 1
fi

if (( page_count < override_page_number )); then
  echo "Output PDF only has ${page_count} pages; cannot replace page ${override_page_number}" >&2
  exit 1
fi

compiled_pages_dir="${build_dir}/compiled_pages"
manual_pages_dir="${build_dir}/manual_pages"
mkdir -p "${compiled_pages_dir}" "${manual_pages_dir}"

pdfseparate "${output_pdf}" "${compiled_pages_dir}/page-%03d.pdf"
pdfseparate -f "${override_page_number}" -l "${override_page_number}" \
  "${manual_pdf}" "${manual_pages_dir}/page-%03d.pdf"

stitched_parts=()
for ((page_index = 1; page_index <= page_count; page_index++)); do
  if (( page_index == override_page_number )); then
    stitched_parts+=("${manual_pages_dir}/page-018.pdf")
  else
    stitched_parts+=("${compiled_pages_dir}/page-$(printf '%03d' "${page_index}").pdf")
  fi
done

pdfunite "${stitched_parts[@]}" "${build_dir}/stitched.pdf"
gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite \
  -sOutputFile="${build_dir}/normalized.pdf" "${build_dir}/stitched.pdf"
mv "${build_dir}/normalized.pdf" "${output_pdf}"

echo "Built and stitched ${output_pdf} with manual page ${override_page_number}."
