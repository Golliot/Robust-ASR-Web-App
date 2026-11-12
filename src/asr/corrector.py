"""Correction d'erreurs optionnelle du notebook (cellule 23), indépendante du modèle.

Elle agit sur le texte produit par le décodage. Chaque mot inconnu du vocabulaire d'entraînement est
remplacé par le mot connu le plus proche, à au plus deux éditions près. À distance égale, le plus
fréquent gagne. Désactivée par défaut (`decoding.use_corrector: false`), comme dans le notebook :
ce module, qui dépend de rapidfuzz, n'est importé que si elle est activée.
"""

from collections import Counter

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein


class WordCorrector:
    def __init__(self, transcripts, min_count=2, max_distance=2):
        counts = Counter(word for text in transcripts for word in text.split())
        self.vocabulary = [word for word, count in counts.most_common() if count >= min_count]
        self.known = set(self.vocabulary)
        self.max_distance = max_distance
        self.corrections = {}

    def correct_word(self, word):
        if word in self.known:
            return word
        if word not in self.corrections:
            match = process.extractOne(
                word, self.vocabulary, scorer=Levenshtein.distance, score_cutoff=self.max_distance
            )
            self.corrections[word] = match[0] if match else word
        return self.corrections[word]

    def __call__(self, text):
        return " ".join(self.correct_word(word) for word in text.split())

