#!/usr/bin/env python3
"""
Task 7D: Retry failed features from task7c (JSON parse errors).
Fixes: max_tokens 512 -> 800, loads existing results and patches in-place.

Usage:
  cd sae_code/
  export ANTHROPIC_API_KEY="sk-ant-..."
  /opt/miniconda3/envs/scanpy-env/bin/python task7d_retry_errors.py
"""

import os, json, time, anthropic

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL   = "claude-haiku-4-5-20251001"

# Features that fell back to prior name due to JSON error
# (identified from task7c output: "Unterminated string" errors)
ERROR_FIDS = [1284, 9, 19, 75, 90, 254, 470, 576, 979, 164, 193, 208,
              763, 1527, 1917, 31, 858, 528, 799, 944]

# Load existing t-test annotations
with open("sae_annotations_rag_ttest.json") as f:
    annotations = json.load(f)
ann_dict = {a["feature_id"]: a for a in annotations}

# Load t-test gene signatures and MyGene cache
with open("gene_signatures_ttest.json") as f:
    raw_t = json.load(f)
ttest_sigs   = {int(k): v["genes"]              for k, v in raw_t.items()}
ct_comps     = {int(k): v["celltype_composition"] for k, v in raw_t.items()}

with open("gene_function_cache.json") as f:
    gene_cache = json.load(f)

with open("task6_cluster_assignments.csv") as f:
    import csv
    reader = csv.DictReader(f)
    cluster_map = {int(row["feature_id"]): int(row["cluster"]) for row in reader}

with open("sae_annotations.json") as f:
    ann_zero = {a["feature_id"]: a for a in json.load(f)}

SYSTEM_PROMPT = """You are an expert haematologist and computational biologist specialising in
haematopoietic differentiation. You are annotating features from a Sparse Autoencoder (SAE)
trained on the BoneMarrowMap human bone marrow atlas. Each feature represents a candidate
Functional Programme (FP) — a coordinated gene expression programme active in a specific
cell state or differentiation stage.

The gene list comes from a t-test comparing top-activating cells vs background.
Use the gene function information to give mechanistically specific annotations.

Respond ONLY as valid JSON with exactly these fields:
{
  "programme_name": "concise mechanistic name (maximum 6 words)",
  "functional_description": "1-2 sentences describing the specific biological mechanism",
  "key_evidence": ["GENE1: reason", "GENE2: reason", "GENE3: reason"],
  "confidence": "HIGH | MEDIUM | LOW",
  "suggested_validation": "one specific experimental approach"
}"""

USER_TEMPLATE = """Feature ID: {fid}
Cluster: {cluster} | Prior: "{prior}"

Cell-type composition (top 200 activating cells):
{ct_str}

Top genes by t-test with functional context:
{gene_context}

Identify the specific Functional Programme. Be concise — keep descriptions to 1-2 sentences."""

assert API_KEY, "Set ANTHROPIC_API_KEY"
client = anthropic.Anthropic(api_key=API_KEY)

print(f"Retrying {len(ERROR_FIDS)} features with max_tokens=800...", flush=True)
n_fixed = 0

for fid in ERROR_FIDS:
    genes = ttest_sigs.get(fid, [])
    if not genes:
        print(f"  F#{fid}: no t-test genes found, skip", flush=True)
        continue

    gene_lines = []
    for g in genes[:12]:  # cap at 12 genes to keep prompt shorter
        info    = gene_cache.get(g, {})
        name    = info.get("name", g)
        summary = info.get("summary", "")
        short   = (summary[:100] + "...") if len(summary) > 100 else summary
        gene_lines.append(f"  {g} ({name}): {short}" if short else f"  {g} ({name})")

    ct  = ct_comps.get(fid, {})
    ct_str = "\n".join(
        f"  {k}: {v}%" for k, v in sorted(ct.items(), key=lambda x: -x[1])
    ) or "  (no dominant cell type)"

    cluster_id = cluster_map.get(fid, "?")
    prior_name = ann_zero.get(fid, {}).get("programme_name", "unknown")

    user_msg = USER_TEMPLATE.format(
        fid          = fid,
        cluster      = cluster_id,
        prior        = prior_name,
        ct_str       = ct_str,
        gene_context = "\n".join(gene_lines),
    )

    try:
        response = client.messages.create(
            model      = MODEL,
            max_tokens = 800,   # was 512 — this was the bug
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())

        new_name = result.get("programme_name", prior_name)
        print(f"  F#{fid:4d}: {prior_name[:30]} -> {new_name}", flush=True)

        # Patch the annotation in place
        if fid in ann_dict:
            ann_dict[fid]["programme_name"]         = new_name
            ann_dict[fid]["functional_description"] = result.get("functional_description", "")
            ann_dict[fid]["key_evidence"]            = result.get("key_evidence", [])
            ann_dict[fid]["confidence"]              = result.get("confidence", "LOW")
            ann_dict[fid]["suggested_validation"]    = result.get("suggested_validation", "")
        n_fixed += 1

    except Exception as e:
        print(f"  F#{fid}: STILL ERROR — {e}", flush=True)

    time.sleep(1.5)

# Save patched file
updated = list(ann_dict.values())
with open("sae_annotations_rag_ttest.json", "w") as f:
    json.dump(updated, f, indent=2)

print(f"\nFixed {n_fixed}/{len(ERROR_FIDS)} errors. File updated in place.", flush=True)
