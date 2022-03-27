from enum import Enum
from sys import intern


class Piece(Enum):
    """A connect four game piece. RED is represented by 'X' and BLACK is represented by 'O'."""
    EMPTY = intern(" ")
    RED = intern("X")
    BLACK = intern("O")

    def __str__(self) -> str:
        return f'{self.name} ({self.value})'

    def __repr__(self) -> str:
        return self.value
