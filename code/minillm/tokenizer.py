import hashlib


class CharacterTokenizer:
    """Explicit ordered alphabet: its order is part of model identity."""

    def __init__(self, alphabet="abcdefgh"):
        if not alphabet or len(set(alphabet)) != len(alphabet):
            raise ValueError("alphabet must be nonempty and contain unique characters")
        self.alphabet = alphabet
        self.lookup = {char: index for index, char in enumerate(alphabet)}

    @property
    def identity(self):
        return hashlib.sha256(("char-v1:" + self.alphabet).encode("utf-8")).hexdigest()

    def encode(self, text):
        return [self.lookup[char] for char in text]

    def decode(self, ids):
        if any(type(i) is not int or not 0 <= i < len(self.alphabet) for i in ids):
            raise ValueError("token outside alphabet")
        return "".join(self.alphabet[i] for i in ids)
