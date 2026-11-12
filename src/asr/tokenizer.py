"""Tokenizer caractère du notebook (cellule 5), repris à l'identique.

Les transcriptions de LibriSpeech n'utilisent que A-Z, l'apostrophe et l'espace. Le blank CTC occupe
l'indice 0, et le dernier indice sert à la fois de début et de fin de séquence pour le décodeur.
"""

import torch


class CharTokenizer:
    def __init__(self):
        self.chars = " 'ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.char_to_id = {char: index + 1 for index, char in enumerate(self.chars)}
        self.blank = 0
        self.sos = self.eos = len(self.chars) + 1
        self.vocab_size = len(self.chars) + 2

    def encode(self, text):
        return torch.tensor([self.char_to_id[char] for char in text], dtype=torch.long)

    def decode(self, ids):
        return "".join(self.chars[index - 1] for index in ids if self.blank < index < self.eos)
