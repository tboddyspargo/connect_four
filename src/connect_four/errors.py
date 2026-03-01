class Error(Exception):
    """A base error class for the ConnectFour module."""

    def __init__(self, message="ConnectFour: Unknown Exception occurred.") -> None:
        self.message = message

    def __str__(self) -> str:
        return self.message

class OutOfBoundsError(Error):
    """A piece was played outside of the bounds of the game board."""


class InvalidPieceError(Error):
    """This game piece cannot be used in this manner"""


class InvalidInsertError(Error):
    """The player tried to insert a piece improperly."""


class InvalidRemoveError(Error):
    """The player tried to remove a piece improperly."""


class InvalidPlayersError(Error):
    """Invalid number of players provided."""


class BoardFullError(Error):
    """Invalid number of players provided."""
