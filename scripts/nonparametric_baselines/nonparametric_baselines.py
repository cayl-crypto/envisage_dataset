import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import KernelDensity, NearestNeighbors


VAL_PATH = "captions_with_uncertainty_envisage_val.json"
TEST_PATH = "captions_with_uncertainty_envisage_test.json"

OUT_TEST_PATH = "captions_with_kde_knn_uncertainty_envisage_test.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_sequence_nlls(data):
    return np.array([x["sequence_avg_nll"] for x in data], dtype=float)


def get_token_nlls(data):
    values = []
    for item in data:
        for tok in item.get("tokenwise", []):
            values.append(tok["nll"])
    return np.array(values, dtype=float)


def assign_knn_bands(cal_scores, test_scores, k=5):
    """
    kNN uncertainty:
    larger distance to calibration scores = higher uncertainty.
    """
    cal_scores = np.asarray(cal_scores, dtype=float).reshape(-1, 1)
    test_scores = np.asarray(test_scores, dtype=float).reshape(-1, 1)

    k = min(k, len(cal_scores))

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(cal_scores)

    cal_dist, _ = knn.kneighbors(cal_scores)
    test_dist, _ = knn.kneighbors(test_scores)

    cal_unc = cal_dist.mean(axis=1)
    test_unc = test_dist.mean(axis=1)

    q68, q95, q997 = np.percentile(cal_unc, [68, 95, 99.7])

    bands = []
    for u in test_unc:
        if u <= q68:
            bands.append("low")
        elif u <= q95:
            bands.append("med")
        elif u <= q997:
            bands.append("high")
        else:
            bands.append("very_high")

    return (
        bands,
        test_unc.tolist(),
        {
            "q68": float(q68),
            "q95": float(q95),
            "q997": float(q997),
            "k": int(k),
        },
    )


def assign_kde_bands(cal_scores, test_scores, bandwidth=0.3):
    """
    KDE uncertainty:
    lower density under calibration distribution = higher uncertainty.
    """
    cal_scores = np.asarray(cal_scores, dtype=float).reshape(-1, 1)
    test_scores = np.asarray(test_scores, dtype=float).reshape(-1, 1)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(cal_scores)

    cal_log_density = kde.score_samples(cal_scores)
    test_log_density = kde.score_samples(test_scores)

    # Since high density means low uncertainty, use lower-tail thresholds.
    # Approximate same band proportions as conformal:
    # low = top 68% density
    # med = next 27%
    # high = next 4.7%
    # very_high = lowest 0.3%
    d68, d95, d997 = np.percentile(cal_log_density, [32, 5, 0.3])

    bands = []
    for d in test_log_density:
        if d >= d68:
            bands.append("low")
        elif d >= d95:
            bands.append("med")
        elif d >= d997:
            bands.append("high")
        else:
            bands.append("very_high")

    return (
        bands,
        test_log_density.tolist(),
        {
            "d68": float(d68),
            "d95": float(d95),
            "d997": float(d997),
            "bandwidth": float(bandwidth),
        },
    )


def attach_sequence_bands(test_data, kde_bands, kde_scores, knn_bands, knn_scores):
    for item, kb, ks, nb, ns in zip(
        test_data, kde_bands, kde_scores, knn_bands, knn_scores
    ):
        item["sequence_uncertainty_band_kde"] = kb
        item["sequence_kde_log_density"] = ks
        item["sequence_uncertainty_band_knn"] = nb
        item["sequence_knn_distance"] = ns


def attach_token_bands(test_data, kde_bands, kde_scores, knn_bands, knn_scores):
    idx = 0
    for item in test_data:
        for tok in item.get("tokenwise", []):
            tok["band_kde"] = kde_bands[idx]
            tok["kde_log_density"] = kde_scores[idx]
            tok["band_knn"] = knn_bands[idx]
            tok["knn_distance"] = knn_scores[idx]
            idx += 1


def count_bands(data, key):
    counts = {"low": 0, "med": 0, "high": 0, "very_high": 0}
    for item in data:
        b = item[key]
        counts[b] = counts.get(b, 0) + 1
    return counts


def count_token_bands(data, key):
    counts = {"low": 0, "med": 0, "high": 0, "very_high": 0}
    for item in data:
        for tok in item.get("tokenwise", []):
            b = tok[key]
            counts[b] = counts.get(b, 0) + 1
    return counts


def main():
    val_data = load_json(VAL_PATH)
    test_data = load_json(TEST_PATH)

    # -------------------------
    # Caption-level scores
    # -------------------------
    val_seq_nll = get_sequence_nlls(val_data)
    test_seq_nll = get_sequence_nlls(test_data)

    seq_kde_bands, seq_kde_scores, seq_kde_info = assign_kde_bands(
        val_seq_nll,
        test_seq_nll,
        bandwidth=0.3,
    )

    seq_knn_bands, seq_knn_scores, seq_knn_info = assign_knn_bands(
        val_seq_nll,
        test_seq_nll,
        k=5,
    )

    attach_sequence_bands(
        test_data,
        seq_kde_bands,
        seq_kde_scores,
        seq_knn_bands,
        seq_knn_scores,
    )

    # -------------------------
    # Token-level scores
    # -------------------------
    val_token_nll = get_token_nlls(val_data)
    test_token_nll = get_token_nlls(test_data)

    tok_kde_bands, tok_kde_scores, tok_kde_info = assign_kde_bands(
        val_token_nll,
        test_token_nll,
        bandwidth=0.3,
    )

    tok_knn_bands, tok_knn_scores, tok_knn_info = assign_knn_bands(
        val_token_nll,
        test_token_nll,
        k=5,
    )

    attach_token_bands(
        test_data,
        tok_kde_bands,
        tok_kde_scores,
        tok_knn_bands,
        tok_knn_scores,
    )

    save_json(test_data, OUT_TEST_PATH)

    print("Saved:", OUT_TEST_PATH)

    print("\nCaption-level KDE thresholds:")
    print(seq_kde_info)
    print("Caption-level KDE band counts:")
    print(count_bands(test_data, "sequence_uncertainty_band_kde"))

    print("\nCaption-level kNN thresholds:")
    print(seq_knn_info)
    print("Caption-level kNN band counts:")
    print(count_bands(test_data, "sequence_uncertainty_band_knn"))

    print("\nToken-level KDE thresholds:")
    print(tok_kde_info)
    print("Token-level KDE band counts:")
    print(count_token_bands(test_data, "band_kde"))

    print("\nToken-level kNN thresholds:")
    print(tok_knn_info)
    print("Token-level kNN band counts:")
    print(count_token_bands(test_data, "band_knn"))


if __name__ == "__main__":
    main()
