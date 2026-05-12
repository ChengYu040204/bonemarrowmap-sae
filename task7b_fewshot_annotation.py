#!/usr/bin/env python3
"""
Task 7B: Few-Shot Enhanced SAE Feature Annotation
===================================================
Identical pipeline to task7_rag_annotation.py, with one change:
SYSTEM_PROMPT now contains 3 manually curated few-shot examples
drawn from Task 3 human annotations, covering:
  - B-cell proliferative programme (non-plasma-cell)
  - ER translocon sub-programme (most upstream plasma cell step)
  - BM niche survival signals (same cell type, different function)

This tests whether explicit examples improve annotation specificity
beyond what RAG gene-function context alone achieves.

Comparison matrix:
  sae_annotations.json          Zero-shot plain LLM
  sae_annotations_rag.json      Zero-shot + RAG
  sae_annotations_fewshot.json  Few-shot (3 examples) + RAG  <-- this script

Runs on ANOVA top 10 only (features: 799, 73, 75, 8, 56, 24, 1, 7, 6, 16).

Usage:
  cd sae_code/
  export ANTHROPIC_API_KEY="sk-ant-..."
  python task7b_fewshot_annotation.py
"""

import os, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
GENE_SIG_FILE = "gene_signatures.json"
GENE_CACHE    = "gene_function_cache.json"
CLUSTER_CSV   = "task6_cluster_assignments.csv"
ANN_IN        = "sae_annotations.json"
OUTPUT        = "sae_annotations_fewshot.json"

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL   = "claude-opus-4-6"   # highest quality for comparison

# ANOVA top 10 feature IDs (from final model, l1_penalty=5e-4)
TOP10_FIDS = [799, 73, 75, 8, 56, 24, 1, 7, 6, 16]
TOP_CELLS  = 200

# ============================================================
# SYSTEM PROMPT WITH FEW-SHOT EXAMPLES
# Three examples chosen to demonstrate:
#   (1) B-cell proliferation — not all high-F features are plasma cells
#   (2) ER translocon entry  — how to name a specific mechanistic sub-step
#   (3) BM niche survival    — same cell type but entirely different function
# ============================================================
SYSTEM_PROMPT = """You are an expert haematologist and computational biologist specialising in
haematopoietic differentiation. You are annotating features from a Sparse Autoencoder (SAE)
trained on the BoneMarrowMap human bone marrow atlas. Each feature represents a candidate
Functional Programme (FP): a coordinated gene expression programme active in a specific
cell state or differentiation stage.

CRITICAL REQUIREMENT: Be mechanistically specific. Distinguish sub-programmes within broad
categories. Do NOT use generic labels like "Plasma cell immunoglobulin secretion programme"
when the genes indicate a specific sub-step (e.g. UPR initiation, ER translocon entry,
ERAD quality control, mTORC1 coordination, isotype-specific production, or BM niche survival).

EXAMPLES OF EXPECTED ANNOTATION SPECIFICITY:

--- Example 1 ---
Input genes: STMN1, IGLL1, CD79A, TYMS, PCNA
Cell-type composition: Large Pre-B 70.0%, Pro-B Cycling 29.5%
Gene functions:
  STMN1 (stathmin 1): A ubiquitous cytosolic phosphoprotein that plays an important role in the
    regulation of the cell cycle by destabilising microtubules. Overexpressed in proliferating cells.
  IGLL1 (immunoglobulin lambda-like polypeptide 1): Encodes the surrogate light chain component
    lambda5 of the pre-B cell receptor (pre-BCR). Marks the Large Pre-B stage after VDJ rearrangement.
  CD79A: Component of the BCR complex (Ig-alpha); essential for BCR signalling.
  TYMS (thymidylate synthetase): Key enzyme in DNA synthesis; S-phase marker.
  PCNA (proliferating cell nuclear antigen): DNA clamp for replication; canonical S-phase marker.
Expected output:
{
  "programme_name": "Large Pre-B proliferative burst",
  "functional_description": "Captures the rapid mitotic expansion following successful immunoglobulin heavy-chain VDJ rearrangement. IGLL1 marks the Large Pre-B stage via the pre-BCR surrogate light chain; STMN1, TYMS, and PCNA are canonical S-phase and mitotic markers. The near-exclusive Large Pre-B + cycling Pro-B composition confirms this is the proliferative checkpoint between pro-B and small pre-B stages.",
  "key_evidence": ["IGLL1: surrogate light chain of pre-BCR — Large Pre-B stage marker", "STMN1: microtubule destabiliser driving rapid mitosis", "PCNA: S-phase DNA replication clamp", "TYMS: thymidylate synthesis for DNA replication", "CD79A: BCR-complex Ig-alpha signalling subunit"],
  "confidence": "HIGH",
  "suggested_validation": "IGLL1 flow cytometry to confirm Large Pre-B enrichment; EdU incorporation assay to verify active S-phase proliferation"
}

--- Example 2 ---
Input genes: XBP1, MZB1, IGKC, SSR4, SEC61B
Cell-type composition: Plasma Cell 99.5%
Gene functions:
  XBP1 (X-box binding protein 1): Master transcription factor of the unfolded protein response (UPR);
    drives ER expansion during plasma cell differentiation.
  MZB1 (marginal zone B and B1 cell-specific protein): ER-localised protein that regulates IgM
    secretion and calcium homeostasis in plasma cells.
  IGKC (immunoglobulin kappa constant): Kappa light chain constant region.
  SSR4 (signal sequence receptor subunit 4): TRAP-delta subunit of the translocon-associated protein
    complex; facilitates co-translational import of secretory proteins into the ER lumen.
  SEC61B: Subunit of the SEC61 translocon channel through which nascent polypeptides enter the ER.
Expected output:
{
  "programme_name": "ER translocon co-translational Ig import",
  "functional_description": "Captures the most upstream step of the plasma cell secretory pipeline: co-translational entry of nascent immunoglobulin chains into the ER lumen via the SEC61/SSR4 translocon complex. SEC61B and SSR4 form the channel and associated protein complex through which all secretory proteins must pass before any downstream ER processing (glycosylation, disulfide bonding, chaperone folding, or ERAD) can occur. This is distinct from and upstream of UPR, ERAD, or Ig assembly programmes.",
  "key_evidence": ["SSR4: TRAP-delta translocon-associated protein — gating co-translational import", "SEC61B: SEC61 channel subunit — the entry point into ER lumen", "MZB1: ER calcium regulator supporting IgM secretion homeostasis", "XBP1: UPR master regulator — marks active secretory machinery"],
  "confidence": "HIGH",
  "suggested_validation": "SSR4 knockdown in plasma cell line (e.g. U266) to assess Ig secretion blockade; ER import assay with signal-peptide-fused reporter"
}

--- Example 3 ---
Input genes: XBP1, MZB1, TNFRSF17, SDC1, SLAMF7
Cell-type composition: Plasma Cell 98.5%
Gene functions:
  TNFRSF17 (TNF receptor superfamily member 17, BCMA): Receptor for APRIL and BAFF cytokines
    produced by bone marrow stromal cells; essential for long-lived plasma cell survival in BM niches.
    Primary target of BCMA-directed CAR-T therapy in multiple myeloma.
  SDC1 (syndecan-1, CD138): Plasma cell surface marker; heparan sulphate proteoglycan mediating
    BM stromal cell adhesion.
  SLAMF7 (CS1, CD319, elotuzumab target): NK-cell and plasma cell surface receptor mediating
    BM niche adhesion; targeted by elotuzumab in myeloma therapy.
  XBP1: UPR master regulator — present in all plasma cells.
  MZB1: ER calcium regulator — present in all plasma cells.
Expected output:
{
  "programme_name": "BM niche long-lived plasma cell survival",
  "functional_description": "Captures BM niche-dependent survival signalling rather than immunoglobulin secretion. TNFRSF17 (BCMA) receives APRIL/BAFF survival signals from BM stromal cells to prevent plasma cell apoptosis; SLAMF7 mediates adhesion to the BM niche microenvironment. This feature marks long-lived plasma cells maintained by stromal signals, not active Ig production — distinguished by presence of survival receptors (BCMA, SLAMF7) and absence of Ig synthesis or ER quality-control genes dominating the signature.",
  "key_evidence": ["TNFRSF17 (BCMA): APRIL/BAFF survival receptor — essential for BM niche long-term PC maintenance; CAR-T target in myeloma", "SLAMF7: BM adhesion receptor — elotuzumab clinical target", "SDC1 (CD138): canonical plasma cell BM adhesion marker"],
  "confidence": "HIGH",
  "suggested_validation": "BCMA neutralisation (anti-APRIL) to assess PC viability loss; co-culture with BM stromal cells vs suspension to test niche-dependence"
}
---

Now annotate the new feature below using the same level of mechanistic specificity shown in the examples above.
Respond ONLY as valid JSON with exactly these fields:
{
  "programme_name": "concise mechanistic name (maximum 8 words)",
  "functional_description": "2-3 sentences describing the specific biological mechanism",
  "key_evidence": ["GENE1: mechanistic reason", "GENE2: mechanistic reason", "GENE3: mechanistic reason"],
  "confidence": "HIGH | MEDIUM | LOW",
  "suggested_validation": "one specific experimental approach"
}"""

USER_TEMPLATE = """Feature ID: {feature_id}
Cluster: {cluster} | Prior annotation (too generic — improve this): "{prior_name}"

Cell-type composition of top {top_cells} activating cells:
{celltype_str}

Top genes with functional context from gene databases:
{gene_context}

Provide a mechanistically specific annotation. If the prior annotation was generic, identify
the precise sub-programme from the gene functions above."""


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    import anthropic
    import pandas as pd

    assert API_KEY, "Set ANTHROPIC_API_KEY before running"
    client = anthropic.Anthropic(api_key=API_KEY)

    print("[1/4] Loading cached gene signatures and functions...", flush=True)
    raw_sigs = load_json(GENE_SIG_FILE)
    gene_signatures  = {int(k): v["genes"] for k, v in raw_sigs.items()}
    ct_compositions  = {int(k): v["celltype_composition"] for k, v in raw_sigs.items()}
    gene_cache       = load_json(GENE_CACHE)

    print("[2/4] Loading cluster assignments and prior annotations...", flush=True)
    df_clusters = pd.read_csv(CLUSTER_CSV)
    old_anns    = {a["feature_id"]: a for a in load_json(ANN_IN)}

    print(f"[3/4] Running few-shot annotation on {len(TOP10_FIDS)} features...\n", flush=True)
    results = []

    for i, fid in enumerate(TOP10_FIDS):
        genes = gene_signatures.get(fid, [])
        gene_lines = []
        for g in genes:
            info    = gene_cache.get(g, {})
            name    = info.get("name", g)
            summary = info.get("summary", "")
            if summary:
                short = summary[:150] + "..." if len(summary) > 150 else summary
                gene_lines.append(f"  {g} ({name}): {short}")
            else:
                gene_lines.append(f"  {g} ({name})")

        ct = ct_compositions.get(fid, {})
        ct_str = "\n".join(
            f"  {n}: {p}%"
            for n, p in sorted(ct.items(), key=lambda x: -x[1])
        ) or "  (no dominant cell type)"

        row        = df_clusters[df_clusters["feature_id"] == fid]
        cluster_id = int(row["cluster"].values[0]) if len(row) else "?"
        prior_name = old_anns.get(fid, {}).get("programme_name", "unknown")

        user_msg = USER_TEMPLATE.format(
            feature_id   = fid,
            cluster      = cluster_id,
            prior_name   = prior_name,
            top_cells    = TOP_CELLS,
            celltype_str = ct_str,
            n_genes      = len(genes),
            gene_context = "\n".join(gene_lines),
        )

        try:
            response = client.messages.create(
                model      = MODEL,
                max_tokens = 700,
                system     = SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": user_msg}],
            )
            raw_text = response.content[0].text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            result = json.loads(raw_text.strip())
        except Exception as e:
            print(f"  ERROR F#{fid}: {e}", flush=True)
            result = {
                "programme_name"       : prior_name,
                "functional_description": "",
                "key_evidence"         : [],
                "confidence"           : "LOW",
                "suggested_validation" : "",
            }

        new_name = result.get("programme_name", prior_name)
        print(f"  [{i+1:2d}/10] F#{fid:4d} | {new_name}", flush=True)

        results.append({
            "feature_id"            : fid,
            "cluster"               : cluster_id,
            "anova_rank"            : i + 1,
            "top_genes"             : genes,
            "programme_name"        : new_name,
            "functional_description": result.get("functional_description", ""),
            "key_evidence"          : result.get("key_evidence", []),
            "confidence"            : result.get("confidence", "LOW"),
            "suggested_validation"  : result.get("suggested_validation", ""),
            "prior_programme_name"  : prior_name,
        })

        time.sleep(2.0)

    print(f"\n[4/4] Saving to {OUTPUT}...", flush=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done. {len(results)} features annotated.", flush=True)


if __name__ == "__main__":
    main()
