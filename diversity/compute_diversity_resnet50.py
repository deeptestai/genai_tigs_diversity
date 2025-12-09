#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, pairwise_distances
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import models, transforms
# ================== EDIT THESE PATHS ==================
ICST_LOW_DIR = "/home/vincenzo.riccio/ddpm_diffuser/generated_images/diversity_check_full/cifar10_icst_low"
ICST_HIGH_DIR = "/home/vincenzo.riccio/ddpm_diffuser/generated_images/diversity_check_full/cifar10_icst_high"
SDLLM_LOW_DIR = "/home/vincenzo.riccio/ddpm_diffuser/generated_images/diversity_check_full/cifar10_sdllm_low"
SDLLM_HIGH_DIR = "/home/vincenzo.riccio/ddpm_diffuser/generated_images/diversity_check_full/cifar10_sdllm_high"
OUT_DIR = "./out_icst_sdllm_cifarfull_resnet50"
# =====================================================

SEED = 0
BATCH = 64
EMB_L2 = False
K_LIST = [2, 4, 6, 8, 10, 12]
MAKE_UMAP = True
UMAP_SAMPLE_MAX = 4000

# ---- UMAP (optional) ----
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# -------------------- I/O --------------------
def list_imgs(d):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    s = set()
    for e in exts:
        s.update(glob.glob(str(Path(d) / f"**/*{e}"), recursive=True))
    return sorted(s)


#----------------------------------------------
def build_resnet50(device="cuda"):
    """Load ResNet-50 pretrained on ImageNet-1K and remove classifier."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()  # remove final classification layer
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return model, preprocess



@torch.no_grad()
def embed(paths, model, preprocess, device="cuda", bs=BATCH):
    vecs = []
    for i in range(0, len(paths), bs):
        imgs = []
        for p in paths[i:i + bs]:
            try:
                im = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                imgs.append(im)
            except Exception:
                continue
        if not imgs:
            continue
        x = torch.cat(imgs).to(device)
        z = model(x).float()
        if EMB_L2:
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-9)
        vecs.append(z.cpu().numpy())
    if not vecs:
        return np.zeros((0, 2048), np.float32)
    return np.concatenate(vecs, 0)


# ----------------- Metrics -------------------
def cov_ent_div(labels, K):
    if len(labels) == 0 or K <= 1:
        return 0.0, 0.0, 0.0
    counts = np.bincount(labels, minlength=K).astype(float)
    cov = float((counts > 0).sum() / K)
    p = counts[counts > 0] / counts.sum()
    H = float((-(p * np.log2(p)).sum()) / np.log2(K)) if len(p) else 0.0
    return cov, H, cov * H


def mean_pairwise_cosine(X, max_pairs=50000, seed=SEED):
    n = len(X)
    if n < 2:
        return 0.0
    rng = np.random.RandomState(seed)
    m = min(max_pairs, n * (n - 1) // 2)
    idx = rng.choice(n, size=(2 * m,), replace=True)
    A, B = X[idx[:m]], X[idx[m:]]
    D = pairwise_distances(A, B, metric="cosine", n_jobs=1)
    return float(np.mean(D))


# --------------- UMAP helpers ----------------
def umap_2d(X, seed=SEED, sample_max=UMAP_SAMPLE_MAX):
    N = len(X)
    if N == 0:
        return np.zeros((0, 2), np.float32), np.arange(0)
    idx = np.arange(N)
    if N > sample_max:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(N, sample_max, replace=False))
    reducer = umap.UMAP(n_components=2, random_state=seed)
    Z2 = reducer.fit_transform(X[idx])
    return Z2, idx


# ---------- Plotters ----------
def scatter_clusters_markers(Z2k, labels_sub, dataset_ids_sub, K, out_png, title):
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    cmap = plt.get_cmap("tab20").resampled(max(2, K))
    marker_for_dataset = {0: "o", 1: "^"}
    from matplotlib.lines import Line2D

    for c in range(K):
        color = cmap(c % cmap.N)
        sel_c = (labels_sub == c)
        if not np.any(sel_c):
            continue
        for ds in (0, 1):
            sel = sel_c & (dataset_ids_sub == ds)
            if not np.any(sel):
                continue
            ax.scatter(Z2k[sel, 0], Z2k[sel, 1], s=14, alpha=0.9, edgecolors="none",
                       label=f"C{c}" if ds == 0 else None,
                       color=color, marker=marker_for_dataset[ds])

    cluster_handles = [Line2D([0], [0], marker='o', color='w',
                              label=f"C{k}", markerfacecolor=cmap(k % cmap.N), markersize=6)
                       for k in range(K)]
    dataset_handles = [
        Line2D([0], [0], marker='o', color='black', label='ICST (shape)', markerfacecolor='w', markersize=6),
        Line2D([0], [0], marker='^', color='black', label='SDLLM (shape)', markerfacecolor='w', markersize=6)
    ]
    first_legend = ax.legend(handles=cluster_handles, title="Clusters", ncol=min(4, K), fontsize=8, frameon=True)
    ax.add_artist(first_legend)
    ax.legend(handles=dataset_handles, title="Dataset", fontsize=9, frameon=True, loc="lower right")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close(fig)


# -------------------- MAIN --------------------
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = build_resnet50(device)

    order = ["ICST_LOW", "ICST_HIGH", "SDLLM_LOW", "SDLLM_HIGH"]
    groups = {
        "ICST_LOW": list_imgs(ICST_LOW_DIR),
        "ICST_HIGH": list_imgs(ICST_HIGH_DIR),
        "SDLLM_LOW": list_imgs(SDLLM_LOW_DIR),
        "SDLLM_HIGH": list_imgs(SDLLM_HIGH_DIR),
    }

    Zs, sizes = {}, {}
    for g in order:
        Z = embed(groups[g], model, preprocess, device=device)
        Zs[g] = Z
        sizes[g] = len(Z)
        print(f"[{g}] N={len(Z)}")
        np.save(Path(OUT_DIR) / f"emb_{g}.npy", Z)

    Z_all = np.vstack([Zs[g] for g in order]) if order else np.zeros((0, 512), np.float32)
    breaks = np.cumsum([sizes[g] for g in order])
    idxs = np.split(np.arange(len(Z_all)), breaks[:-1]) if len(Z_all) else [np.array([], dtype=int) for _ in order]

    dataset_ids = np.empty(len(Z_all), dtype=int)
    start = 0
    for gi, g in enumerate(order):
        n = sizes[g]
        dataset_ids[start:start + n] = 0 if "ICST" in g else 1
        start += n

    rows = []

    for K in K_LIST:
        if len(Z_all) == 0 or K <= 1 or K > len(Z_all):
            print(f"[warn] Skipping K={K} (invalid for N={len(Z_all)})")
            continue

        km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z_all)
        labels = km.labels_

        metric = ("euclidean" if EMB_L2 else "cosine")
        sil_union = silhouette_score(Z_all, labels, metric=metric) if len(np.unique(labels)) > 1 else 0.0

        # Compute ARI metrics
        ds_label = np.concatenate([
            np.full(len(idxs[0]), 0),
            np.full(len(idxs[1]), 0),
            np.full(len(idxs[2]), 1),
            np.full(len(idxs[3]), 1),
        ])
        pert_label = np.concatenate([
            np.full(len(idxs[0]), 0),
            np.full(len(idxs[1]), 1),
            np.full(len(idxs[2]), 0),
            np.full(len(idxs[3]), 1),
        ])
        ari_dataset = adjusted_rand_score(labels, ds_label)
        ari_perturb = adjusted_rand_score(labels, pert_label)

        # Per-group metrics
        for g, idg in zip(order, idxs):
            lab_g = labels[idg] if len(idg) else np.array([], dtype=int)
            cov, H, Dv = cov_ent_div(lab_g, K)
            mpd = mean_pairwise_cosine(Zs[g]) if sizes[g] else 0.0
            rows.append({
                "K": K, "group": g, "n": sizes[g],
                "coverage": cov, "entropy": H, "diversity": Dv,
                "mean_pairwise_cosine": mpd,
                "silhouette_union": sil_union,
                "ARI_cluster_vs_dataset(ICST_vs_SDLLM)": ari_dataset,
                "ARI_cluster_vs_perturb(LOW_vs_HIGH)": ari_perturb,
            })

        # --- Overall (concatenated ICST/SDLLM)
        Z_icst_all = np.concatenate([Zs["ICST_LOW"], Zs["ICST_HIGH"]])
        Z_sdllm_all = np.concatenate([Zs["SDLLM_LOW"], Zs["SDLLM_HIGH"]])
        labels_icst_all = np.concatenate([labels[idxs[0]], labels[idxs[1]]])
        labels_sdllm_all = np.concatenate([labels[idxs[2]], labels[idxs[3]]])

        for name, Z_all_g, lab_all in [
            ("ICST_OVERALL", Z_icst_all, labels_icst_all),
            ("SDLLM_OVERALL", Z_sdllm_all, labels_sdllm_all)
        ]:
            cov, H, Dv = cov_ent_div(lab_all, K)
            mpd = mean_pairwise_cosine(Z_all_g)
            rows.append({
                "K": K, "group": name, "n": len(Z_all_g),
                "coverage": cov, "entropy": H, "diversity": Dv,
                "mean_pairwise_cosine": mpd,
                "silhouette_union": sil_union,
                "ARI_cluster_vs_dataset(ICST_vs_SDLLM)": ari_dataset,
                "ARI_cluster_vs_perturb(LOW_vs_HIGH)": ari_perturb,
            })

        # --- Cluster-based UMAP visualizations ---
        if MAKE_UMAP and HAS_UMAP:
            # Low vs Low clusters
            if len(Zs["ICST_LOW"]) > 0 and len(Zs["SDLLM_LOW"]) > 0:
                Z_low = np.vstack([Zs["ICST_LOW"], Zs["SDLLM_LOW"]])
                ds_low = np.array([0]*len(Zs["ICST_LOW"]) + [1]*len(Zs["SDLLM_LOW"]))
                km_low = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z_low)
                labels_low = km_low.labels_
                Z2_low, idx_sub = umap_2d(Z_low, seed=SEED)
                scatter_clusters_markers(
                    Z2_low, labels_low[idx_sub], ds_low[idx_sub],
                    K=K,
                    out_png=str(Path(OUT_DIR)/f"umap_low_vs_low_clusters_K{K}.png"),
                    title=f"UMAP — Low vs Low Clusters (K={K})"
                )

            # High vs High clusters
            if len(Zs["ICST_HIGH"]) > 0 and len(Zs["SDLLM_HIGH"]) > 0:
                Z_high = np.vstack([Zs["ICST_HIGH"], Zs["SDLLM_HIGH"]])
                ds_high = np.array([0]*len(Zs["ICST_HIGH"]) + [1]*len(Zs["SDLLM_HIGH"]))
                km_high = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z_high)
                labels_high = km_high.labels_
                Z2_high, idx_sub = umap_2d(Z_high, seed=SEED)
                scatter_clusters_markers(
                    Z2_high, labels_high[idx_sub], ds_high[idx_sub],
                    K=K,
                    out_png=str(Path(OUT_DIR)/f"umap_high_vs_high_clusters_K{K}.png"),
                    title=f"UMAP — High vs High Clusters (K={K})"
                )

            # Overall vs Overall clusters
            Z_overall = np.vstack([
                Zs["ICST_LOW"], Zs["ICST_HIGH"],
                Zs["SDLLM_LOW"], Zs["SDLLM_HIGH"]
            ])
            ds_overall = np.array(
                [0]*(len(Zs["ICST_LOW"])+len(Zs["ICST_HIGH"])) +
                [1]*(len(Zs["SDLLM_LOW"])+len(Zs["SDLLM_HIGH"]))
            )
            km_overall = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z_overall)
            labels_overall = km_overall.labels_
            Z2_overall, idx_sub = umap_2d(Z_overall, seed=SEED)
            scatter_clusters_markers(
                Z2_overall, labels_overall[idx_sub], ds_overall[idx_sub],
                K=K,
                out_png=str(Path(OUT_DIR)/f"umap_overall_vs_overall_clusters_K{K}.png"),
                title=f"UMAP — Overall vs Overall Clusters (K={K})"
            )

            # Global clusters
            if len(Z_all) > 0:
                Z2k, idx_sub = umap_2d(Z_all, seed=SEED, sample_max=UMAP_SAMPLE_MAX)
                labels_sub = labels[idx_sub]
                dataset_ids_sub = dataset_ids[idx_sub]
                scatter_clusters_markers(
                    Z2k, labels_sub=labels_sub, dataset_ids_sub=dataset_ids_sub,
                    K=K,
                    out_png=str(Path(OUT_DIR)/f"umap_global_clusters_K{K}.png"),
                    title=f"UMAP — Global Clusters (K={K}, colors=clusters, shapes=dataset)"
                )

        np.save(Path(OUT_DIR) / f"labels_union_K{K}.npy", labels)
        print(f"[K={K}] silhouette={sil_union:.3f} | ARI(ds)={ari_dataset:.3f} | ARI(pert)={ari_perturb:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(Path(OUT_DIR) / "diversity_union_ICST_SDLLM_low_high_overall_over_K.csv", index=False)
    print("\n[done] wrote results & plots to:", OUT_DIR)


if __name__ == "__main__":
    main()
