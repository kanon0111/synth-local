from typing import List, Dict
from collections import Counter
import re
try:
    from langdetect import detect
except Exception:
    detect = None

PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?:\+?\d{1,3}[-\s]?)?(?:\(?\d{2,4}\)?[-\s]?)?\d{3,4}[-\s]?\d{3,4}")
}
TOXIC_WORDS = {"kill","hate","stupid","idiot","死ね","殺す","バカ"}

def language_match_rate(texts: List[str], target_lang: str) -> float:
    if detect is None: return 1.0
    ok = 0
    for t in texts:
        try:
            if detect(t) == target_lang:
                ok += 1
        except Exception:
            pass
    return ok / max(1, len(texts))

def basic_length_stats(texts: List[str]) -> Dict[str, float]:
    lens = [len(t) for t in texts]
    return {"avg": (sum(lens)/len(lens) if lens else 0), "min": (min(lens) if lens else 0), "max": (max(lens) if lens else 0)}

def ngram_dup_rate(texts: List[str], n: int = 5) -> float:
    seen = Counter(); total = 0
    for t in texts:
        toks = t.split()
        for i in range(0, max(len(toks)-n+1, 0)):
            seen[tuple(toks[i:i+n])] += 1; total += 1
    if total == 0: return 0.0
    dup = sum(c-1 for c in seen.values() if c > 1)
    return dup / total

def toxicity_rate(texts: List[str]) -> float:
    return sum(1 for t in texts if any(w in t.lower() for w in TOXIC_WORDS)) / max(1, len(texts))

def pii_rate(texts: List[str]) -> float:
    return sum(1 for t in texts if any(p.search(t) for p in PII_PATTERNS.values())) / max(1, len(texts))

def summarize_quality(responses: List[str], target_lang: str="ja") -> Dict:
    return {
        "language_match": language_match_rate(responses, target_lang),
        "length": basic_length_stats(responses),
        "dup_5gram_rate": ngram_dup_rate(responses, 5),
        "toxicity_rate": toxicity_rate(responses),
        "pii_rate": pii_rate(responses),
    }

def pass_fail(report: Dict) -> Dict[str, bool]:
    return {
        "language_match_ok": report.get("language_match", 1.0) >= 0.95,
        "dup_ok": report.get("dup_5gram_rate", 0.0) <= 0.05,
        "toxicity_ok": report.get("toxicity_rate", 0.0) <= 0.01,
        "pii_ok": report.get("pii_rate", 0.0) == 0.0,
    }
