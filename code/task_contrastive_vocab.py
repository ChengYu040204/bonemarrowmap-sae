"""
Vocabulary-level contrastive analysis for Cluster 2 plasma cell features.

Problem: Global t-test gives all 41 features the same top genes (SSR4/FKBP11/PRDX4/SEC11C etc.
appearing in 100% of features), making LLM name them identically.

Solution: For each feature, remove genes shared with >SHARE_THRESH fraction of other Cluster 2
features. The residual genes represent what is unique to that sub-programme.
Falls back to >50% threshold if <MIN_UNIQUE genes remain after filtering.
"""
import json, os, time
import pandas as pd
import numpy as np
from collections import Counter
import anthropic

BASE = "/sessions/lucid-inspiring-wright/mnt/sae_code"
TTEST_JSON   = f"{BASE}/gene_signatures_ttest.json"
CLUSTER_CSV  = f"{BASE}/task6_cluster_assignments.csv"
CACHE_JSON   = f"{BASE}/gene_function_cache.json"
RAG_JSON     = f"{BASE}/sae_annotations_rag.json"
OUT_SIGS     = f"{BASE}/gene_signatures_contrastive.json"
OUT_ANN      = f"{BASE}/sae_annotations_contrastive.json"

SHARE_THRESH = 0.30   # remove genes in >30% of other C2 features
FALLBACK_T   = 0.50   # fallback to 50% if <MIN_UNIQUE genes remain
MIN_UNIQUE   = 8
TOP_GENES    = 25
TOP_CONTEXT  = 15

SYSTEM_PROMPT = """You are an expert computational biologist specialising in haematopoiesis and plasma cell biology.

You are given genes that UNIQUELY characterise one SAE (Sparse Autoencoder) feature within a cluster of 41 plasma cell features.
The shared plasma cell identity genes (SSR4, FKBP11, PRDX4, SEC11C, MZB1, XBP1 etc.) have been removed.
The remaining genes therefore reflect the SPECIFIC sub-programme this feature represents within plasma cell differentiation.

Focus your annotation on the molecular mechanism these unique genes imply.

Respond ONLY as valid JSON with these fields:
  programme_name      (6 words max — specific sub-programme, NOT generic "plasma cell secretion")
  functional_description (2-3 sentences focusing on the distinguishing mechanism)
  key_evidence        (list of 5 strings, format: "GENE: why this gene defines the sub-programme")
  confidence          ("HIGH", "MEDIUM", or "LOW")
  sub_programme_layer (one of: "ER_translocon", "UPR_stress", "IgG_isotype", "IgA_isotype",
                       "IgM_IgA_polymerisation", "IgKappa_secretion", "IgLambda_secretion",
                       "ERAD_quality_control", "survival_signalling", "oxidative_proteostasis", "other")"""

USER_TEMPLATE = """SAE Feature ID: {feat_id}
ANOVA Rank (among all B-cell features): {rank}
Cluster: 2 (Plasma cell terminal differentiation, 41 features)

SHARED PLASMA CELL BACKGROUND (present in >50% of all Cluster 2 features, context only):
  SSR4, FKBP11, PRDX4, SEC11C, MZB1, XBP1, PLPP5, FKBP2

UNIQUE GENES for Feature {feat_id}:
(Appear in <{thresh_pct}% of other Cluster 2 features; these DEFINE this sub-programme)
{gene_block}

GENE FUNCTION CONTEXT:
{ctx_block}

This is a plasma cell feature. Identify the SPECIFIC sub-programme based on the unique genes.
Focus on: Ig isotype (IgG/IgA/IgM/kappa/lambda), secretion mechanism, or ER quality control sub-type."""

def build_contrastive_sigs(c2_ids, ttest_sigs, thresh=SHARE_THRESH):
    genes_per_feat = {}
    for fid in c2_ids:
        entry = ttest_sigs.get(fid, {})
        genes = entry.get('genes', [])
        genes_per_feat[fid] = genes[:30]

    gene_count = Counter()
    for g_list in genes_per_feat.values():
        for g in g_list:
            gene_count[g] += 1

    n = len(c2_ids)
    results = {}
    for fid in c2_ids:
        raw = genes_per_feat.get(fid, [])
        unique = [g for g in raw if gene_count[g] / n <= thresh]
        if len(unique) < MIN_UNIQUE:
            unique = [g for g in raw if gene_count[g] / n <= FALLBACK_T]
            used_thresh = FALLBACK_T
        else:
            used_thresh = thresh
        results[fid] = {'genes': unique[:TOP_GENES], 'thresh': used_thresh,
                        'n_unique': len(unique), 'n_raw': len(raw)}
    return results, gene_count

def build_ctx(genes, cache):
    lines = []
    for g in genes[:TOP_CONTEXT]:
        desc = cache.get(g, {})
        summary = ""
        if isinstance(desc, dict):
            summary = desc.get("summary", "") or desc.get("name", "")
        elif isinstance(desc, str):
            summary = desc
        if summary:
            lines.append(f"  {g}: {summary[:120]}")
        else:
            lines.append(f"  {g}: (no annotation)")
    return "\n".join(lines)

def annotate_feature(client, feat_id, rank, genes, cache, thresh):
    gene_block = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(genes))
    ctx_block  = build_ctx(genes, cache)
    thresh_pct = int(thresh * 100)
    prompt = USER_TEMPLATE.format(
        feat_id=feat_id, rank=rank, thresh_pct=thresh_pct,
        gene_block=gene_block, ctx_block=ctx_block
    )
    resp = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=900,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)

def main():
    df = pd.read_csv(CLUSTER_CSV)
    c2_ids_int = df[df['cluster']==2]['feature_id'].tolist()
    c2_ids_str = [str(x) for x in c2_ids_int]
    print(f"Cluster 2: {len(c2_ids_str)} features", flush=True)

    with open(TTEST_JSON) as f:
        ttest_sigs = json.load(f)
    with open(CACHE_JSON) as f:
        gene_cache = json.load(f)
    with open(RAG_JSON) as f:
        rag_list = json.load(f)
    rag_map = {e['feature_id']: e for e in rag_list}

    # Build contrastive signatures
    print("\n=== Building vocabulary-level contrastive signatures ===", flush=True)
    sigs, gene_count = build_contrastive_sigs(c2_ids_str, ttest_sigs)

    for fid in c2_ids_str:
        info = sigs[fid]
        print(f"  F#{fid}: {info['n_unique']} unique genes (thresh={info['thresh']:.0%}): {info['genes'][:6]}", flush=True)

    with open(OUT_SIGS, 'w') as f:
        json.dump(sigs, f, indent=2)
    print(f"\nSaved {len(sigs)} contrastive signatures to {OUT_SIGS}", flush=True)

    # Pairwise Jaccard BEFORE
    all_orig = [set(ttest_sigs.get(fid, {}).get('genes', [])[:30]) for fid in c2_ids_str]
    j_before = []
    for i in range(len(all_orig)):
        for j in range(i+1, len(all_orig)):
            a, b = all_orig[i], all_orig[j]
            if a | b:
                j_before.append(len(a & b) / len(a | b))
    print(f"Mean pairwise Jaccard BEFORE contrastive: {np.mean(j_before):.3f} (global t-test baseline = 0.592)")

    all_contr = [set(sigs[fid]['genes']) for fid in c2_ids_str]
    j_after = []
    for i in range(len(all_contr)):
        for j in range(i+1, len(all_contr)):
            a, b = all_contr[i], all_contr[j]
            if a | b:
                j_after.append(len(a & b) / len(a | b))
    print(f"Mean pairwise Jaccard AFTER  contrastive: {np.mean(j_after):.3f}")
    print(f"  (vs RAG/Pearson baseline: 0.421)")

    # Claude API annotation
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not set. Export key and re-run.", flush=True)
        return

    existing = {}
    if os.path.exists(OUT_ANN):
        with open(OUT_ANN) as f:
            ex = json.load(f)
        existing = {e['feature_id']: e for e in ex}
        print(f"\nResuming: {len(existing)}/{len(c2_ids_int)} already annotated", flush=True)

    client = anthropic.Anthropic(api_key=api_key)
    results = dict(existing)

    print(f"\n=== Claude API annotation ({len(c2_ids_int)} features) ===", flush=True)
    for i, (fid_int, fid_str) in enumerate(zip(c2_ids_int, c2_ids_str)):
        if fid_int in results:
            continue
        info = sigs[fid_str]
        genes = info['genes']
        thresh = info['thresh']
        rag_entry = rag_map.get(fid_int, {})
        rank = rag_entry.get('anova_rank', 999)

        print(f"  [{i+1}/{len(c2_ids_int)}] F#{fid_int} (rank {rank}, {len(genes)} unique genes)...", end=' ', flush=True)
        time.sleep(2)
        try:
            ann = annotate_feature(client, fid_int, rank, genes, gene_cache, thresh)
            ann['feature_id'] = fid_int
            ann['anova_rank'] = rank
            ann['top_genes'] = genes
            ann['method'] = 'contrastive_vocab'
            ann['n_unique'] = info['n_unique']
            results[fid_int] = ann
            print(f"{ann.get('programme_name','?')} [{ann.get('confidence','?')}] layer={ann.get('sub_programme_layer','?')}", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            results[fid_int] = {
                'feature_id': fid_int, 'programme_name': 'ERROR',
                'confidence': 'LOW', 'method': 'contrastive_vocab',
                'top_genes': genes, 'error': str(e)
            }

        if (i+1) % 5 == 0:
            with open(OUT_ANN, 'w') as f:
                json.dump(list(results.values()), f, indent=2)

    out_list = list(results.values())
    with open(OUT_ANN, 'w') as f:
        json.dump(out_list, f, indent=2)
    print(f"\nSaved {len(out_list)} annotations to {OUT_ANN}", flush=True)

    # Final evaluation
    print("\n=== Evaluation ===", flush=True)
    names = [r.get('programme_name','') for r in out_list if r.get('programme_name','') not in ('ERROR','')]
    unique_names = set(names)
    layers = [r.get('sub_programme_layer','') for r in out_list]
    layer_counts = Counter(layers)

    print(f"Cluster 2 unique programme names (contrastive): {len(unique_names)}")
    print(f"  vs RAG/Pearson baseline:  31 unique names")
    print(f"  vs global t-test baseline: 25 unique names")
    print(f"\nSub-programme layer distribution:")
    for layer, cnt in layer_counts.most_common():
        print(f"  {layer}: {cnt}")
    print(f"\nAll unique programme names:")
    for n in sorted(unique_names):
        print(f"  - {n}")

if __name__ == '__main__':
    main()
