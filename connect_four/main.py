from .connect_four import ConnectFour
from .piece import Piece


def main() -> None:
    """main() is only expected to be used during development to verify functionality."""
    # fmt: off
    board = [
        [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
        [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.RED, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
        [Piece.EMPTY, Piece.EMPTY, Piece.RED, Piece.BLACK, Piece.RED, Piece.EMPTY, Piece.EMPTY],
        [Piece.EMPTY, Piece.EMPTY, Piece.BLACK, Piece.RED, Piece.BLACK, Piece.EMPTY, Piece.EMPTY],
        [Piece.EMPTY, Piece.EMPTY, Piece.RED, Piece.BLACK, Piece.BLACK, Piece.EMPTY, Piece.EMPTY],
        [Piece.EMPTY, Piece.BLACK, Piece.BLACK, Piece.RED, Piece.RED, Piece.EMPTY, Piece.EMPTY]
    ]
    # fmt: on

    game = ConnectFour.new(
        players=0,
        human_player=Piece.RED,
        # board = board,
        log_level="DEBUG",
    )

    game.play(12)


if __name__ == "__main__":
    main()
