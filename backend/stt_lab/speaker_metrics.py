"""Dependency-light, deterministic metrics for speaker-attributed transcripts."""
from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

from .metrics import normalize


def _hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Minimum-cost rectangular assignment (rows or columns may be unmatched)."""
    if not cost or not cost[0]:
        return []
    original_rows, original_cols = len(cost), len(cost[0])
    transposed = original_rows > original_cols
    matrix = [list(row) for row in cost]
    if transposed:
        matrix = [list(row) for row in zip(*matrix)]
    n, m = len(matrix), len(matrix[0])
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = float("inf"), 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j], way[j] = current, j0
                if minimum[j] < delta:
                    delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]
    if transposed:
        pairs = [(column, row) for row, column in pairs]
    return [(row, column) for row, column in pairs
            if row < original_rows and column < original_cols]


def _word_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, left in enumerate(reference, 1):
        current = [row]
        for column, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (left != right)))
        previous = current
    return previous[-1]


def _active(segments, point, key="speaker_id"):
    return {segment.get(key) for segment in segments
            if segment["start_ms"] <= point < segment["end_ms"] and segment.get(key) is not None}


def _intervals(reference, hypothesis, duration_ms, collar_ms, skip_overlap):
    boundaries = {0.0, float(duration_ms)}
    reference_edges = []
    for segment in [*reference, *hypothesis]:
        boundaries.update((float(segment["start_ms"]), float(segment["end_ms"])))
    for segment in reference:
        reference_edges.extend((float(segment["start_ms"]), float(segment["end_ms"])))
        for edge in (segment["start_ms"], segment["end_ms"]):
            boundaries.add(max(0.0, edge - collar_ms))
            boundaries.add(min(float(duration_ms), edge + collar_ms))
    ordered = sorted(value for value in boundaries if 0 <= value <= duration_ms)
    result = []
    for start, end in pairwise(ordered):
        if end <= start:
            continue
        midpoint = (start + end) / 2
        refs = _active(reference, midpoint)
        if any(abs(midpoint - edge) < collar_ms for edge in reference_edges):
            continue
        if skip_overlap and len(refs) > 1:
            continue
        result.append((end - start, refs, _active(hypothesis, midpoint, "cluster_id")))
    return result


def diarization(reference, hypothesis, duration_ms, *, collar_ms=250, skip_overlap=True):
    intervals = _intervals(reference, hypothesis, duration_ms, collar_ms, skip_overlap)
    ref_labels = sorted({label for _, refs, _ in intervals for label in refs})
    hyp_labels = sorted({label for _, _, hyps in intervals for label in hyps})
    overlap = [[0.0 for _ in ref_labels] for _ in hyp_labels]
    for duration, refs, hyps in intervals:
        for hyp in hyps:
            for ref in refs:
                overlap[hyp_labels.index(hyp)][ref_labels.index(ref)] += duration
    maximum = max((value for row in overlap for value in row), default=0.0)
    pairs = _hungarian([[maximum - value for value in row] for row in overlap])
    mapping = {hyp_labels[row]: ref_labels[column] for row, column in pairs}
    scored = miss = false_alarm = confusion = correct = 0.0
    for duration, refs, hyps in intervals:
        matched = sum(1 for hyp in hyps if mapping.get(hyp) in refs)
        ref_count, hyp_count = len(refs), len(hyps)
        scored += ref_count * duration
        correct += matched * duration
        miss += max(0, ref_count - hyp_count) * duration
        false_alarm += max(0, hyp_count - ref_count) * duration
        confusion += (min(ref_count, hyp_count) - matched) * duration
    der = (miss + false_alarm + confusion) / scored if scored else None
    jaccards = []
    for ref in ref_labels:
        mapped_hyp = next((hyp for hyp, target in mapping.items() if target == ref), None)
        intersection = union = 0.0
        for duration, refs, hyps in intervals:
            in_ref = ref in refs
            in_hyp = mapped_hyp in hyps if mapped_hyp is not None else False
            intersection += duration if in_ref and in_hyp else 0.0
            union += duration if in_ref or in_hyp else 0.0
        jaccards.append(1 - intersection / union if union else 0.0)
    return {
        "der": der,
        "miss": miss / scored if scored else None,
        "false_alarm": false_alarm / scored if scored else None,
        "confusion": confusion / scored if scored else None,
        "jer": sum(jaccards) / len(jaccards) if jaccards else None,
        "mapping": mapping,
    }


def concatenated_permutation_wer(reference, hypothesis):
    ref_text = defaultdict(list)
    hyp_text = defaultdict(list)
    for segment in sorted(reference, key=lambda item: item["start_ms"]):
        ref_text[segment["speaker_id"]].extend(normalize(segment.get("text", "")).split())
    for segment in sorted(hypothesis, key=lambda item: item["start_ms"]):
        hyp_text[segment["cluster_id"]].extend(normalize(segment.get("text", "")).split())
    refs, hyps = list(ref_text.values()), list(hyp_text.values())
    size = max(len(refs), len(hyps), 1)
    cost = []
    for row in range(size):
        left = refs[row] if row < len(refs) else []
        cost.append([_word_distance(left, hyps[column] if column < len(hyps) else [])
                     for column in range(size)])
    errors = sum(cost[row][column] for row, column in _hungarian(cost))
    words = sum(map(len, refs))
    return errors / words if words else None


def named_speaker_wer(reference, hypothesis, speakers):
    ref_known, hyp_known = defaultdict(list), defaultdict(list)
    ref_unknown, hyp_unknown = defaultdict(list), defaultdict(list)
    for segment in sorted(reference, key=lambda item: item["start_ms"]):
        words = normalize(segment.get("text", "")).split()
        profile_id = speakers[segment["speaker_id"]].get("profile_id")
        (ref_known[profile_id] if profile_id else
         ref_unknown[segment["speaker_id"]]).extend(words)
    for segment in sorted(hypothesis, key=lambda item: item["start_ms"]):
        words = normalize(segment.get("text", "")).split()
        profile_id = segment.get("profile_id")
        (hyp_known[profile_id] if profile_id else
         hyp_unknown[segment["cluster_id"]]).extend(words)
    errors = 0
    for ident in set(ref_known) | set(hyp_known):
        errors += _word_distance(ref_known.get(ident, []), hyp_known.get(ident, []))
    unknown_refs, unknown_hyps = list(ref_unknown.values()), list(hyp_unknown.values())
    size = max(len(unknown_refs), len(unknown_hyps))
    if size:
        cost = [
            [_word_distance(unknown_refs[row] if row < len(unknown_refs) else [],
                            unknown_hyps[column] if column < len(unknown_hyps) else [])
             for column in range(size)]
            for row in range(size)
        ]
        errors += sum(cost[row][column] for row, column in _hungarian(cost))
    words = sum(len(value) for value in [*ref_known.values(), *ref_unknown.values()])
    return errors / words if words else None


def _identity(segment, speakers, *, hypothesis=False):
    if hypothesis:
        return segment.get("profile_id") or "__unknown__"
    speaker = speakers.get(segment["speaker_id"], {})
    return speaker.get("profile_id") or "__unknown__"


def identity_scores(reference, hypothesis, speakers, duration_ms):
    boundaries = sorted({0.0, float(duration_ms),
                         *(float(s[key]) for s in [*reference, *hypothesis]
                           for key in ("start_ms", "end_ms"))})
    totals = defaultdict(float)
    correct = 0.0
    total = 0.0
    labels = set()
    for start, end in pairwise(boundaries):
        midpoint = (start + end) / 2
        refs = [s for s in reference if s["start_ms"] <= midpoint < s["end_ms"]]
        hyps = [s for s in hypothesis if s["start_ms"] <= midpoint < s["end_ms"]]
        if len(refs) != 1:
            continue
        ref = _identity(refs[0], speakers)
        hyp = (_identity(hyps[0], speakers, hypothesis=True)
               if len(hyps) == 1 else "__missing__")
        weight = end - start
        totals[(ref, hyp)] += weight
        labels.update((ref, hyp))
        total += weight
        correct += weight if ref == hyp else 0.0
    per_label = {}
    for label in labels:
        tp = totals[(label, label)]
        fp = sum(value for (ref, hyp), value in totals.items() if hyp == label and ref != label)
        fn = sum(value for (ref, hyp), value in totals.items() if ref == label and hyp != label)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else 0.0)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}
    f1s = [value["f1"] for value in per_label.values()]
    return {"accuracy": correct / total if total else None,
            "macro_f1": sum(f1s) / len(f1s) if f1s else None,
            "unknown": per_label.get("__unknown__")}


def evaluate(reference_data, hypothesis, duration_seconds):
    if not reference_data:
        return {"available": False}
    reference = reference_data["segments"]
    speakers = {item["id"]: item for item in reference_data["speakers"]}
    duration_ms = duration_seconds * 1000
    no_overlap = diarization(reference, hypothesis, duration_ms, skip_overlap=True)
    with_overlap = diarization(reference, hypothesis, duration_ms, skip_overlap=False)
    return {
        "available": True,
        "der": no_overlap,
        "der_with_overlap": with_overlap,
        "cpwer": concatenated_permutation_wer(reference, hypothesis),
        "speaker_attributed_wer": named_speaker_wer(
            reference, hypothesis, speakers),
        "identity": identity_scores(reference, hypothesis, speakers, duration_ms),
        "speaker_count_error": abs(len({s["speaker_id"] for s in reference}) -
                                   len({s["cluster_id"] for s in hypothesis})),
    }
