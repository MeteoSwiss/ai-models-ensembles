#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Submit the OUTSTANDING baseline eval modules after the all-in-one main job
# timed out at 12 h. Strategy: one sbatch per (model, module) for the heavy
# modules, plus a single bundled job for the light ones.
#
# Modules:
#   light (det + ssim)  : 1 bundled job, 4 streams (one per model), sequential per stream
#   multivariate        : 4 jobs (one per model)
#   probabilistic       : 4 jobs (one per model)
#   ets                 : 4 jobs (members mode, with quantile precompute fix)
#   fss                 : 4 jobs (members mode, with quantile precompute fix)
#   energy_spectra      : per-model only if missing (aifsens, ifs_ens)
#
# Usage:
#   bash scripts/evaluate_baselines_remaining.sh                 # submit all 19
#   bash scripts/evaluate_baselines_remaining.sh dry             # show what would run
# ---------------------------------------------------------------------------
set -euo pipefail

STORE="/capstor/store/cscs/swissai/a122/sadamov/ai-models-ensembles"
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$STORE/baseline_eval_logs"
PY="$SRC_DIR/.venv/bin/python3"

PARTITION="${PARTITION:-normal}"

MODELS="atlas fcn3 aifsens ifs_ens"

# Which models still need energy_spectra. Update by hand if status changes.
ENERGY_SPECTRA_REDO="aifsens ifs_ens"

mkdir -p "$LOG_DIR"

DRY="${1:-}"
[[ "$DRY" == "dry" ]] && DRY=1 || DRY=0

# ---------------------------------------------------------------------------
# Generate a module-subset config from an existing main/etsfss template.
# ---------------------------------------------------------------------------
gen_subset_config() {
    local model=$1
    local subset_name=$2   # tag for the output file
    local modules_yaml=$3  # python literal: {"deterministic": True, ...}
    local template=$4      # "main" or "etsfss"

    local src_cfg dst_cfg
    if [[ "$template" == "etsfss" ]]; then
        src_cfg="$STORE/baselines/${model}/eval_etsfss_config.yaml"
    else
        src_cfg="$STORE/baselines/${model}/eval_main_config.yaml"
    fi
    dst_cfg="$STORE/baselines/${model}/eval_${subset_name}_config.yaml"

    $PY -c "
import yaml
with open('$src_cfg') as f: c = yaml.safe_load(f)
c['modules'] = $modules_yaml
with open('$dst_cfg', 'w') as f: yaml.safe_dump(c, f, sort_keys=False)
"
    echo "$dst_cfg"
}

# All-false template (set specific modules True via merging)
MOD_ALL_FALSE='{"maps": False, "histograms": False, "wd_kde": False, "energy_spectra": False, "vertical_profiles": False, "deterministic": False, "ets": False, "fss": False, "probabilistic": False, "ssim": False, "multivariate": False}'

# ---------------------------------------------------------------------------
# Helper: submit a single per-module sbatch.
# ---------------------------------------------------------------------------
submit_single() {
    local job_tag=$1
    local cfg_path=$2
    local time_limit=$3

    if [[ "$DRY" == "1" ]]; then
        echo "  [DRY] sbatch $job_tag  cfg=$(basename $cfg_path)  time=$time_limit"
        return
    fi

    sbatch --parsable \
        --job-name="$job_tag" \
        --partition="$PARTITION" \
        --account=a122 \
        --nodes=1 --ntasks=1 --cpus-per-task=144 --mem=444G --time="$time_limit" \
        --output="$LOG_DIR/${job_tag}_%j.out" \
        --error="$LOG_DIR/${job_tag}_%j.err" \
        --wrap="source ${SRC_DIR}/.venv/bin/activate && \
            python -m swissclim_evaluations.cli --config '${cfg_path}'"
}

# ---------------------------------------------------------------------------
# 1. Light bundle: deterministic + ssim for all 4 models, 4-stream sbatch.
# ---------------------------------------------------------------------------
submit_light_bundle() {
    echo "=== light bundle (det + ssim, all 4 models, 4 streams) ==="

    # Per-model config combining deterministic + ssim
    local mods_light='{"maps": False, "histograms": False, "wd_kde": False, "energy_spectra": False, "vertical_profiles": False, "deterministic": True, "ets": False, "fss": False, "probabilistic": False, "ssim": True, "multivariate": False}'
    for model in $MODELS; do
        gen_subset_config "$model" "light" "$mods_light" "main" > /dev/null
    done

    local helper="$LOG_DIR/eval_baseline_light_bundle.sh"
    cat > "$helper" <<EOF
#!/bin/bash
set -e
source "${SRC_DIR}/.venv/bin/activate"
EOF

    for model in $MODELS; do
        cat >> "$helper" <<EOF
(
    cfg="$STORE/baselines/${model}/eval_light_config.yaml"
    log="$LOG_DIR/eval_baseline_light_${model}.log"
    echo "[${model}] starting light (det+ssim)"
    python -m swissclim_evaluations.cli --config "\$cfg" > "\$log" 2>&1 && echo "[${model}] done" || echo "[${model}] FAILED"
) &
EOF
    done
    echo "wait" >> "$helper"
    chmod +x "$helper"

    if [[ "$DRY" == "1" ]]; then
        echo "  [DRY] sbatch eval_baseline_light_bundle  helper=$helper"
        return
    fi

    sbatch --parsable \
        --job-name="eval_baseline_light_bundle" \
        --partition="$PARTITION" \
        --account=a122 \
        --nodes=1 --ntasks=1 --cpus-per-task=144 --mem=444G --time="12:00:00" \
        --output="$LOG_DIR/eval_baseline_light_bundle_%j.out" \
        --error="$LOG_DIR/eval_baseline_light_bundle_%j.err" \
        --wrap="bash $helper"
}

# ---------------------------------------------------------------------------
# 2/3. Per-module per-model: multivariate, probabilistic.
# ---------------------------------------------------------------------------
submit_per_model_single_module() {
    local module=$1     # e.g. "multivariate"
    local time_limit=$2

    echo "=== ${module} (one job per model) ==="
    local mods="${MOD_ALL_FALSE}"
    # Set the requested module to True via python (safer than sed)
    local mods_py="${mods//False, \"${module}\": False/False, \"${module}\": True}"
    # Simpler: regenerate via python with module overridden
    for model in $MODELS; do
        local cfg
        cfg=$($PY -c "
import yaml
template = '$STORE/baselines/${model}/eval_main_config.yaml'
with open(template) as f: c = yaml.safe_load(f)
all_off = {k: False for k in c['modules']}
all_off['${module}'] = True
c['modules'] = all_off
dst = '$STORE/baselines/${model}/eval_${module}_only_config.yaml'
with open(dst, 'w') as f: yaml.safe_dump(c, f, sort_keys=False)
print(dst)
")
        submit_single "eval_baseline_${model}_${module}" "$cfg" "$time_limit"
    done
}

# ---------------------------------------------------------------------------
# 4/5. ETS / FSS: stride 24h, members mode. Use the etsfss template, restrict
# to one module each, one job per model.
# ---------------------------------------------------------------------------
submit_per_model_etsfss() {
    local module=$1   # "ets" or "fss"
    local time_limit=$2

    echo "=== ${module} (members mode, one job per model) ==="
    for model in $MODELS; do
        local cfg
        cfg=$($PY -c "
import yaml
template = '$STORE/baselines/${model}/eval_etsfss_config.yaml'
with open(template) as f: c = yaml.safe_load(f)
all_off = {k: False for k in c['modules']}
all_off['${module}'] = True
c['modules'] = all_off
dst = '$STORE/baselines/${model}/eval_${module}_only_config.yaml'
with open(dst, 'w') as f: yaml.safe_dump(c, f, sort_keys=False)
print(dst)
")
        submit_single "eval_baseline_${model}_${module}" "$cfg" "$time_limit"
    done
}

# ---------------------------------------------------------------------------
# 6. Energy spectra fixup: only models flagged as incomplete.
# ---------------------------------------------------------------------------
submit_energy_spectra_fixup() {
    echo "=== energy_spectra fixup (only models in ENERGY_SPECTRA_REDO) ==="
    for model in $ENERGY_SPECTRA_REDO; do
        local cfg
        cfg=$($PY -c "
import yaml
template = '$STORE/baselines/${model}/eval_main_config.yaml'
with open(template) as f: c = yaml.safe_load(f)
all_off = {k: False for k in c['modules']}
all_off['energy_spectra'] = True
c['modules'] = all_off
dst = '$STORE/baselines/${model}/eval_energy_spectra_only_config.yaml'
with open(dst, 'w') as f: yaml.safe_dump(c, f, sort_keys=False)
print(dst)
")
        submit_single "eval_baseline_${model}_energy_spectra" "$cfg" "12:00:00"
    done
}

# ---------------------------------------------------------------------------

submit_light_bundle
submit_per_model_single_module multivariate "12:00:00"
submit_per_model_single_module probabilistic "12:00:00"
submit_per_model_etsfss ets "06:00:00"
submit_per_model_etsfss fss "06:00:00"
submit_energy_spectra_fixup

echo ""
echo "Logs: $LOG_DIR/"
