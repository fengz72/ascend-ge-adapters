"""Named business-output comparisons only."""
import numpy as np


def numeric(reference, candidate, policy):
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {"passed": False, "reason": "nonfinite"}
    mode = policy.get("policy", "numeric")
    if mode == "exact":
        ok = reference.dtype == candidate.dtype and np.array_equal(reference, candidate)
        return {"passed": bool(ok)}
    if mode == "topk":
        k = policy.get("top_k", 1)
        if reference.ndim < 1 or not 1 <= k <= reference.shape[-1]:
            raise ValueError("invalid top_k")
        a = np.argsort(-reference, axis=-1, kind="stable")[..., :k]
        b = np.argsort(-candidate, axis=-1, kind="stable")[..., :k]
        return {"passed": bool(np.array_equal(a, b))}
    a = reference.astype(np.float64)
    b = candidate.astype(np.float64)
    if a.size == 0:
        raise ValueError("empty numeric output")
    rows = [(a.reshape(-1), b.reshape(-1))]
    if policy.get("scope", "per_sample") == "per_sample" and a.ndim:
        rows = [(x.reshape(-1), y.reshape(-1)) for x, y in zip(a, b)]
    metrics = []
    for x, y in rows:
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        diff = np.abs(x - y)
        if nx == 0:
            close = bool(np.all(diff <= policy.get("atol", 1e-5)))
            cosine, l2 = (1.0, 0.0) if close else (0.0, None)
            passed = close
        else:
            cosine = float(np.dot(x, y) / (nx * ny)) if ny else 0.0
            l2 = float(np.linalg.norm(x - y) / nx)
            passed = cosine >= policy.get("cosine_min", 0.999) and l2 <= policy.get("relative_l2_max", 0.02)
        if "rtol" in policy or "atol" in policy:
            passed = passed and bool(np.allclose(x, y, rtol=policy.get("rtol", 0), atol=policy.get("atol", 0)))
        if "max_abs_error" in policy:
            passed = passed and bool(diff.max() <= policy["max_abs_error"])
        metrics.append({"passed": bool(passed), "cosine": cosine, "relative_l2": l2, "max_abs_error": float(diff.max())})
    return {"passed": all(x["passed"] for x in metrics), "failed_samples": [i for i, x in enumerate(metrics) if not x["passed"]], "samples": metrics}


def compare(reference, candidate, policies):
    if reference["alignment"] != candidate["alignment"]:
        raise ValueError("request/column alignment mismatch")
    if set(reference["outputs"]) != set(candidate["outputs"]) or set(policies) != set(reference["outputs"]):
        raise ValueError("missing, extra or unconfigured output")
    outputs = {name: numeric(np.asarray(reference["outputs"][name]), np.asarray(candidate["outputs"][name]), policies[name]) for name in policies}
    return {"passed": all(x["passed"] for x in outputs.values()), "outputs": outputs}
