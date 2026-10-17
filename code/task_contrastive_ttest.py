"""
Contrastive t-test for Cluster 2 (plasma cell) SAE features.
For each feature A: compare top-200 cells vs within-cluster background
(union of top-200 cells from all other Cluster 2 features, minus feature A cells).
This filters out the shared plasma cell signal and extracts sub-programme-specific genes.
"""
import pickle, json, os, time
import numpy as np
import pandas as pd
from scipy import stats
import anthropic

SAE_PKL   = "/sessions/lucid-inspiring-wright/mnt/sae_code/sae_results_all.pkl"
H5AD_PATH = "/sessions/lucid-inspiring-wright/mnt/sae_code/../../Papers' Dataset/BoneMarrowMap_Unsorted_scVI.h5ad"
CLUSTER_CSV = "/sessions/lucid-inspiring-wright/mnt/sae_code/task6_cluster_assignments.csv"
RAG_JSON  = "/sessions/lucid-inspiring-wright/mnt/sae_code/sae_annotations_rag.json"
OUT_SIGS  = "/sessions/lucid-inspiring-wright/mnt/sae_code/gene_signatures_contrastive.json"
OUT_ANN   = "/sessions/lucid-inspiring-wright/mnt/sae_code/sae_annotations_contrastive.json"
CACHE_JSON = "/sessions/lucid-inspiring-wright/mnt/sae_code/gene_function_cache.json"

TOP_K = 200
TOP_GENES = 30
SYSTEM_PROMPT = """You are an expert computational biologist specialising in haematopoiesis and B-cell/plasma cell differentiation.
You will be given a gene list derived by comparing ONE specific SAE (Sparse Autoencoder) feature's high-activation cells
against a within-cluster background of OTHER plasma cell features. The gene list therefore highlights what is UNIQUE to
this particular sub-programme, not shared plasma cell identity. Focus on the molecular mechanism that distinguishes this
sub-programme within plasma cell differentiation.

Respond ONLY as valid JSON with fields:
  programme_name      (6 words max, specific sub-programme name)
  functional_description (2-3 sentences, focus on distinguishing mechanism)
  key_evidence        (list of 5 strings "GENE: reason this gene distinguishes this sub-programme")
  confidence          ("HIGH", "MEDIUM", or "LOW")
  sub_programme_layer ("ER_translocon", "UPR_stress", "Ig_assembly", "mTORC1_translation", "ERAD_quality_control",
                       "survival_signalling", "Ig_secretion", "oxidative_proteostasis", or "other")"""

USER_TEMPLATE = """SAE Feature ID: {feat_id}
ANOVA Rank (among all B-cell features): {rank}
Cluster: 2 (Plasma cell / B-cell terminal differentiation)

CONTRASTIVE TOP {n_genes} GENES:
(These genes distinguish Feature {feat_id} from other plasma cell features in Cluster 2)
{gene_block}

GENE FUNCTION CONTEXT:
{ctx_block}

Identify the specific sub-programme for this feature within plasma cell biology."""

def load_sae():
    print("Loading SAE features...", flush=True)
    with open(SAE_PKL, "rb") as f:
        d = pickle.load(f)
    raw = d[1]
    X = raw.detach().cpu().numpy() if hasattr(raw, 'detach') else np.array(raw)
    print(f"  SAE matrix: {X.shape}", flush=True)
    return X

def load_genes():
    import anndata
    print("Loading gene expression...", flush=True)
    adata = anndata.read_h5ad(H5AD_PATH)
    X_genes = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    gene_names = list(adata.var_names)
    print(f"  Gene matrix: {X_genes.shape}", flush=True)
    return X_genes, gene_names

def contrastive_genes(feat_id, c2_ids, X_sae, X_genes, gene_names):
    """t-test: feature A top-K cells vs within-cluster background."""
    feat_col = X_sae[:, feat_id]
    top_a = set(np.argsort(feat_col)[-TOP_K:].tolist())

    # within-cluster background: union of other features' top-K, minus top_a
    bg = set()
    for fid in c2_ids:
        if fid == feat_id:
            continue
        col = X_sae[:, fid]
        bg.update(np.argsort(col)[-TOP_K:].tolist())
    bg -= top_a

    if len(bg) < 20:
        print(f"  Feature {feat_id}: background too small ({len(bg)}), skipping", flush=True)
        return []

    idx_a  = list(top_a)
    idx_bg = list(bg)
    expr_a  = X_genes[idx_a, :]
    expr_bg = X_genes[idx_bg, :]

    t_stats, _ = stats.ttest_ind(expr_a, expr_bg, axis=0, equal_var=False)
    t_clean = np.where(np.isnan(t_stats), -np.inf, t_stats)
    top_idx = np.argsort(t_clean)[::-1][:TOP_GENES]
    return [gene_names[i] for i in top_idx]

def build_ctx(genes, cache):
    lines = []
    for g in genes[:15]:
        desc = cache.get(g, {})
        summary = ""
        if isinstance(desc, dict):
            summary = desc.get("summary", "") or desc.get("name", "")
        elif isinstance(desc, str):
            summary = desc
        if summary:
            summary = summary[:120]
            lines.append(f"  {g}: {summary}")
        else:
            lines.append(f"  {g}: (no annotation available)")
    return "\n".join(lines)

def annotate_feature(client, feat_id, rank, genes, cache):
    gene_block = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(genes[:TOP_GENES]))
    ctx_block  = build_ctx(genes, cache)
    prompt = USER_TEMPLATE.format(
        feat_id=feat_id, rank=rank, n_genes=len(genes),
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
    # Load data
    X_sae  = load_sae()
    X_genes, gene_names = load_genes()

    df = pd.read_csv(CLUSTER_CSV)
    c2_ids = df[df['cluster']==2]['feature_id'].tolist()
    print(f"Cluster 2 features: {len(c2_ids)}", flush=True)

    # Load gene function cache
    with open(CACHE_JSON) as f:
        gene_cache = json.load(f)
    print(f"Gene cache entries: {len(gene_cache)}", flush=True)

    # Load RAG annotations for rank lookup
    with open(RAG_JSON) as f:
        rag_list = json.load(f)
    rag_map = {e['feature_id']: e for e in rag_list}

    # Step 1: Compute contrastive gene signatures
    if os.path.exists(OUT_SIGS):
        print("Loading cached contrastive signatures...", flush=True)
        with open(OUT_SIGS) as f:
            sigs = json.load(f)
    else:
        print("\n=== Computing contrastive gene signatures ===", flush=True)
        sigs = {}
        for feat_id in c2_ids:
            genes = contrastive_genes(feat_id, c2_ids, X_sae, X_genes, gene_names)
            sigs[str(feat_id)] = genes
            print(f"  F#{feat_id}: {genes[:8]}...", flush=True)
        with open(OUT_SIGS, 'w') as f:
            json.dump(sigs, f, indent=2)
        print(f"Saved {len(sigs)} signatures to {OUT_SIGS}", flush=True)

    # Step 2: Claude API annotation
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", flush=True)
        return

    existing = {}
    if os.path.exists(OUT_ANN):
        with open(OUT_ANN) as f:
            existing_list = json.load(f)
        existing = {e['feature_id']: e for e in existing_list}
        print(f"Resuming: {len(existing)}/{len(c2_ids)} already done", flush=True)

    client = anthropic.Anthropic(api_key=api_key)
    results = dict(existing)

    for i, feat_id in enumerate(c2_ids):
        if feat_id in results:
            continue
        genes = sigs.get(str(feat_id), [])
        if not genes:
            print(f"  [{i+1}/{len(c2_ids)}] F#{feat_id}: no genes, skipping", flush=True)
            continue

        rag_entry = rag_map.get(feat_id, {})
        rank = rag_entry.get('anova_rank', 999)

        print(f"  [{i+1}/{len(c2_ids)}] F#{feat_id} (rank {rank})...", end=' ', flush=True)
        time.sleep(2)
        try:
            ann = annotate_feature(client, feat_id, rank, genes, gene_cache)
            ann['feature_id'] = feat_id
            ann['anova_rank'] = rank
            ann['top_genes'] = genes[:TOP_GENES]
            ann['method'] = 'contrastive_ttest'
            results[feat_id] = ann
            print(f"{ann.get('programme_name','?')} [{ann.get('confidence','?')}]", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            results[feat_id] = {
                'feature_id': feat_id, 'programme_name': 'ERROR',
                'confidence': 'LOW', 'method': 'contrastive_ttest',
                'top_genes': genes[:TOP_GENES], 'error': str(e)
            }

        # Save checkpoint every 5 features
        if (i+1) % 5 == 0:
            out_list = list(results.values())
            with open(OUT_ANN, 'w') as f:
                json.dump(out_list, f, indent=2)

    # Final save
    out_list = list(results.values())
    with open(OUT_ANN, 'w') as f:
        json.dump(out_list, f, indent=2)
    print(f"\nSaved {len(out_list)} annotations to {OUT_ANN}", flush=True)

    # Step 3: Evaluate unique programme names
    print("\n=== Evaluation ===", flush=True)
    names = [r.get('programme_name','') for r in out_list if r.get('programme_name','') not in ('ERROR','')]
    unique = set(names)
    print(f"Cluster 2 unique programme names (contrastive): {len(unique)}", flush=True)
    print(f"  vs RAG baseline:    31 unique names", flush=True)
    print(f"  vs t-test baseline: 25 unique names", flush=True)
    for n in sorted(unique):
        print(f"  - {n}", flush=True)

    # Pairwise Jaccard on top_genes
    feat_genes = {r['feature_id']: set(r.get('top_genes',[])) for r in out_list}
    ids = list(feat_genes.keys())
    jaccards = []
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = feat_genes[ids[i]], feat_genes[ids[j]]
            j_score = len(a&b)/len(a|b) if a|b else 0
            jaccards.append(j_score)
    mean_j = np.mean(jaccards) if jaccards else 0
    print(f"\nMean pairwise Jaccard (contrastive genes): {mean_j:.3f}", flush=True)
    print(f"  vs RAG (Pearson) baseline: 0.421", flush=True)
    print(f"  vs global t-test baseline: 0.592", flush=True)
    print("(Lower = more distinct gene signatures = better discrimination)", flush=True)

if __name__ == '__main__':
    main()
