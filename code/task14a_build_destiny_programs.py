# task14a: Step B v2 Phase 1+2 构建谱系命运基因程序 / Build lineage destiny gene programs
#
# 目的 / Why:
#   从各 lineage-specific SAE 的 gene_signatures 中, 用 celltype_composition
#   (top-200 高激活细胞的真实细胞类型构成) 筛出 "成熟终点主导" 的 feature,
#   聚合成每个谱系的 destiny gene set (命运基因集).
#
# 为什么这样做 / Rationale:
#   Step B v1 用 LLM 生成的 programme name 做 TF-IDF 聚类来筛 feature, 与真实激活脱节,
#   且后续在 SAE 激活空间打分时 erythroid 触发 OOD (分布外) 放大, 排名完全失真.
#   v2 改为: (1) feature 选择基于真实激活构成 (in-distribution, 零 OOD),
#            (2) 只输出基因集, 打分留到表达空间 (task14b 的 score_genes).
#
# 本脚本特性 / Note:
#   纯 JSON 操作, 不依赖 scanpy / torch / 161K h5ad, 沙箱与本机均可运行.
#   产物 destiny_gene_sets.json 供 task14b 在 scanpy-env 中打分使用.

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

# 每个 program: (来源文件, 成熟终点细胞类型列表)
# program -> (source gene_signatures file, mature endpoint cell types)
PROGRAMS = {
    "CD14_Mono": ("gene_signatures_mono_routeB.json", ["CD14 Mono"]),
    "CD16_Mono": ("gene_signatures_mono_routeB.json", ["CD16 Mono"]),
    "cDC1":      ("gene_signatures_dc_routeB.json",   ["cDC1"]),
    "cDC2":      ("gene_signatures_dc_routeB.json",   ["cDC2"]),
    "pDC":       ("gene_signatures_dc_routeB.json",   ["pDC"]),
    "T_cell":    ("gene_signatures_tcnk_routeB.json", [
                    "CD4 Naive", "CD4 Central Memory", "CD4 Effector Memory",
                    "CD4 Regulatory", "CD8 Naive", "CD8 Central Memory",
                    "CD8 Effector Memory 1", "CD8 Effector Memory 2",
                    "T Proliferating"]),
    "NK":        ("gene_signatures_tcnk_routeB.json", ["NK", "NK CD56high"]),
    "Mature_B":  ("gene_signatures_bcell_ttest.json", ["Mature B"]),
    "Plasma":    ("gene_signatures_bcell_ttest.json", ["Plasma Cell"]),
}

# canonical markers 仅用于 sanity check, 不参与筛选 / known markers for sanity only
MARKERS = {
    "CD14_Mono": ["CD14", "LYZ", "S100A8", "S100A9", "FCN1", "VCAN", "CSF1R"],
    "CD16_Mono": ["FCGR3A", "CDKN1C", "MS4A7", "LST1", "CSF1R", "LYN"],
    "cDC1":      ["CLEC9A", "XCR1", "BATF3", "IRF8", "CADM1", "C1ORF54"],
    "cDC2":      ["CD1C", "FCER1A", "CLEC10A", "IRF4", "CD1E", "FCGR2B"],
    "pDC":       ["IRF7", "LILRA4", "GZMB", "IL3RA", "TCF4", "CLEC4C", "JCHAIN"],
    "T_cell":    ["CD3D", "CD3E", "CD8A", "CD4", "IL7R", "CCL5", "TRAC", "GZMK"],
    "NK":        ["NCAM1", "NKG7", "GNLY", "KLRD1", "PRF1", "GZMB", "KLRF1"],
    "Mature_B":  ["MS4A1", "CD79A", "CD79B", "BANK1", "IGHD", "TCL1A", "CD19", "CD22"],
    "Plasma":    ["MZB1", "XBP1", "JCHAIN", "IGKC", "SDC1", "PRDM1", "DERL3", "TNFRSF17"],
}

TOP_GENES_PER_FEATURE = 15    # 每个 feature 取前多少基因
ENDPOINT_THRESHOLD = 50.0     # 成熟终点占 top-200 的百分比阈值
THRESHOLD_FALLBACK = [50.0, 40.0, 30.0]  # 选中过少时逐级放宽
MIN_FEATURES_FOR_DENOISE = 4  # 选中 feature 数 >= 此值才做 "基因出现 >= 2 次" 去噪
MAX_GENES = 60                # 每个 program 基因集上限


def endpoint_purity(comp, endpoints):
    """成熟终点细胞类型在该 feature top-200 中的累计占比 (百分比)."""
    return sum(float(comp.get(ct, 0.0)) for ct in endpoints)


def select_features(sig, endpoints):
    """返回 (选中 feature_id 列表按纯度降序, 实际使用阈值)."""
    for thr in THRESHOLD_FALLBACK:
        chosen = []
        for fid, v in sig.items():
            p = endpoint_purity(v.get("celltype_composition", {}), endpoints)
            if p >= thr:
                chosen.append((fid, p))
        if len(chosen) >= 3:
            chosen.sort(key=lambda x: -x[1])
            return chosen, thr
    # 仍然太少: 退回取纯度最高的前 5 个
    allf = [(fid, endpoint_purity(v.get("celltype_composition", {}), endpoints))
            for fid, v in sig.items()]
    allf.sort(key=lambda x: -x[1])
    return allf[:5], -1.0


def build_gene_set(sig, chosen_fids):
    """聚合选中 feature 的 top 基因, 出现频次去噪后截断."""
    cnt = Counter()
    per_feat = {}
    for fid in chosen_fids:
        genes = sig[fid].get("genes", [])[:TOP_GENES_PER_FEATURE]
        genes = [g for g in genes if isinstance(g, str) and g and g.lower() != "nan"]
        per_feat[fid] = genes
        cnt.update(set(genes))
    n = len(chosen_fids)
    if n >= MIN_FEATURES_FOR_DENOISE:
        kept = [g for g, c in cnt.items() if c >= 2]
        if len(kept) < 10:                # 去噪后过少则退回全集
            kept = list(cnt.keys())
    else:
        kept = list(cnt.keys())
    kept.sort(key=lambda g: (-cnt[g], g))  # 按频次降序, 同频按字母
    return kept[:MAX_GENES], cnt, per_feat


def main():
    cache = {}
    out = {}
    print("=" * 78)
    print("Step B v2 Phase 1+2: destiny gene-program construction / 命运基因程序构建")
    print("=" * 78)

    for prog, (fn, endpoints) in PROGRAMS.items():
        if fn not in cache:
            cache[fn] = json.load(open(os.path.join(HERE, fn)))
        sig = cache[fn]

        chosen, thr = select_features(sig, endpoints)
        fids = [c[0] for c in chosen]
        genes, cnt, per_feat = build_gene_set(sig, fids)
        purities = [c[1] for c in chosen]
        mean_purity = round(sum(purities) / len(purities), 1) if purities else 0.0

        mk = MARKERS.get(prog, [])
        hits = [g for g in mk if g in set(genes)]

        out[prog] = {
            "source_file": fn,
            "endpoint_celltypes": endpoints,
            "threshold_used": thr,
            "n_features": len(fids),
            "feature_ids": fids,
            "mean_endpoint_purity": mean_purity,
            "n_genes": len(genes),
            "genes": genes,
            "marker_hits": hits,
        }

        print(f"\n>> {prog}   (source: {fn})")
        print(f"   endpoints: {endpoints}")
        print(f"   features selected: {len(fids)}  "
              f"(threshold={thr}%, mean endpoint purity={mean_purity}%)")
        print(f"   gene set size: {len(genes)}")
        print(f"   top 15 genes: {genes[:15]}")
        print(f"   canonical marker hits ({len(hits)}/{len(mk)}): {hits}")

    dest = os.path.join(HERE, "destiny_gene_sets.json")
    json.dump(out, open(dest, "w"), indent=2)
    print("\n" + "=" * 78)
    print(f"Saved: {dest}")
    print("Programs:", list(out.keys()))


if __name__ == "__main__":
    main()
