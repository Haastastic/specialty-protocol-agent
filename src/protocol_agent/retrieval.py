"""TF-IDF retrieval over protocol pack sections. Replaces 'train on millions of interactions'
with 'retrieve the right 3 chunks of curated knowledge for this turn.'"""

from dataclasses import dataclass
from pathlib import Path

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    specialty: str
    section: str          # scheduling_rules | payer_logic | faq | escalation_triggers
    text: str


def _flatten_pack(pack: dict) -> list[Chunk]:
    specialty = pack["specialty"]
    chunks = []

    for section in ("scheduling_rules", "payer_logic", "escalation_triggers"):
        for item in pack.get(section, []):
            chunks.append(Chunk(specialty=specialty, section=section, text=item))

    for qa in pack.get("faq", []):
        chunks.append(Chunk(specialty=specialty, section="faq",
                             text=f"Q: {qa['question']} A: {qa['answer']}"))

    return chunks


class ProtocolIndex:
    """Loads every protocol pack in a directory and answers retrieval queries."""

    def __init__(self, protocol_dir: str = "protocols"):
        self.chunks: list[Chunk] = []
        for path in sorted(Path(protocol_dir).glob("*.yaml")):
            with open(path) as f:
                pack = yaml.safe_load(f)
            self.chunks.extend(_flatten_pack(pack))

        self.specialties = sorted({c.specialty for c in self.chunks})
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, query: str, specialty: str | None = None, top_k: int = 4) -> list[Chunk]:
        """Return the top_k most relevant chunks, optionally restricted to one specialty."""
        candidate_idxs = [
            i for i, c in enumerate(self.chunks)
            if specialty is None or c.specialty.lower() == specialty.lower()
        ]
        if not candidate_idxs:
            return []

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix[candidate_idxs]).flatten()

        # Always surface escalation triggers regardless of similarity score — safety-critical,
        # never let retrieval ranking silently drop them.
        ranked = sorted(zip(candidate_idxs, sims), key=lambda x: x[1], reverse=True)
        top = [self.chunks[i] for i, _ in ranked[:top_k]]

        escalation_chunks = [
            self.chunks[i] for i in candidate_idxs
            if self.chunks[i].section == "escalation_triggers" and self.chunks[i] not in top
        ]
        return top + escalation_chunks

    def detect_specialty(self, utterance: str) -> str | None:
        """Cheap heuristic: does the utterance mention a known specialty name directly?"""
        lowered = utterance.lower()
        for specialty in self.specialties:
            if specialty.lower() in lowered:
                return specialty
        return None
