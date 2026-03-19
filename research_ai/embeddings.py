"""Simple local embeddings fallback (bag-of-words) and utilities.

This provides a tiny, dependency-free embedding implementation for prototyping. For production, plug in OpenAI embeddings or sentence-transformers.
"""

import re
import math
from collections import Counter
from typing import Dict, List


class Embeddings:
    def __init__(self):
        pass

    def _tokenize(self, text: str):
        return re.findall(r"\w+", text.lower())

    def embed(self, text: str) -> Dict[str, float]:
        toks = self._tokenize(text)
        c = Counter(toks)
        return dict(c)

    def similarity(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        # cosine similarity for sparse dict vectors
        num = 0.0
        for k, v in a.items():
            if k in b:
                num += v * b[k]
        denom_a = math.sqrt(sum(v * v for v in a.values()))
        denom_b = math.sqrt(sum(v * v for v in b.values()))
        if denom_a == 0 or denom_b == 0:
            return 0.0
        return num / (denom_a * denom_b)

    def batch_embed(self, texts: List[str]) -> List[Dict[str, float]]:
        return [self.embed(t) for t in texts]
