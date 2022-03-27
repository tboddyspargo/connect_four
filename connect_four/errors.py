class Error(Exception):
    """A base error class for the ConnectFour module."""
    def __init__(self, message="ConnectFour: Unknown Exception occurred.") -> None:
        self.message = message

    def __str__(self) -> str:
        return self.message

class OutOfBoundsError(Error):
    """A piece was played outside of the bounds of the game board."""
    pass

class InvalidPieceError(Error):
    """This game piece cannot be used in this manner"""
    pass

class InvalidInsertError(Error):
    """The player tried to insert a piece improperly."""
    pass

class InvalidRemoveError(Error):
    """The player tried to remove a piece improperly."""
    pass

class InvalidPlayersError(Error):
    """Invalid number of players provided."""
    pass

class BoardFullError(Error):
    """Invalid number of players provided."""
    pass
