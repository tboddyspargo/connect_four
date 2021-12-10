from enum import Enum, IntEnum, auto
import random
from copy import deepcopy
from typing import Any


def one_index(n: int) -> int:
    return n + 1


#region Logging
class LogLevel(IntEnum):
    NONE = auto()
    INFO = auto()
    DEBUG = auto()
    VERBOSE = auto()

    def __ge__(self, other) -> bool:
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other) -> bool:
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other) -> bool:
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other) -> bool:
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class Logger:
    def __init__(self, level=LogLevel.NONE):
        self.level = level

    def normal(self, *message):
        print(*message)

    def error(self, *message):
        print('[ERROR]', *message)

    def info(self, *message):
        if self.level >= LogLevel.INFO:
            print('[INFO]', *message)

    def debug(self, *message):
        if self.level >= LogLevel.DEBUG:
            print('[DEBUG]', *message)

    def verbose(self, *message):
        if self.level >= LogLevel.VERBOSE:
            print('[VERBOSE]', *message)


#endregion


#region Error Handling
class Error(Exception):
    """A base error class for the ConnectFour module."""
    def __init__(self, message="ConnectFour: Unknown Exception occurred.") -> None:
        self.message = message

    def __str__(self) -> str:
        return repr(self.message)


class OutOfBoundsError(Error):
    """A piece was played outside of the bounds of the game board."""
    pass


class InvalidPieceError(Error):
    """This game piece cannot be used in this manner"""
    pass


class InvalidInsertError(Error):
    """The player tried to insert a piece improperly."""
    pass


class InvalidPlayersError(Error):
    """Invalid number of players provided."""
    pass


#endregion


class Piece(Enum):
    EMPTY = " "
    RED = "X"
    BLACK = "O"

    def __str__(self) -> str:
        return f' {self.value} '

    def __repr__(self) -> str:
        return f'{self.name} ({self.value})'


# TODO: add class for each slot/position?


class ConnectFour:
    #region Attributes and Printing
    def __init__(self,
                 players=0,
                 board=None,
                 pretend=False,
                 log_level=LogLevel.NONE,
                 rows=6,
                 columns=7) -> None:
        self.winner = Piece.EMPTY
        self.winning_group = []
        self.players = players
        self.rows = rows
        self.next_player = Piece.RED
        self.columns = columns
        self.log = Logger(log_level)
        self.pretend = pretend
        self.board = board or [[Piece.EMPTY for c in range(columns)]
                               for r in range(rows)]

    @classmethod
    def new(cls, **kwargs) -> any:
        instance = cls(**kwargs)
        if 'players' not in kwargs:
            instance.players = instance.get_number_players()
        return instance

    @staticmethod
    def lowest_row(positions) -> int:
      result = -1
      for r,c in positions:
        if r > result:
          result = r
      return result

    @staticmethod
    def opponent(piece) -> Piece:
        return Piece.RED if piece is Piece.BLACK else Piece.BLACK

    def highlighted_str(self, highlight_positions=[]):
        row_div = "|---------------------------|"
        result = []
        for r in range(self.rows):
            row = []
            for c in range(self.columns):
                piece_str = str(self.board[r][c])
                if (r, c) in highlight_positions:
                    piece_str = '|' + piece_str[1] + '|'
                row.append(piece_str)
            result.append("|" + "|".join(row) + "|")
            # result.append(row_div)

        result.append(row_div)
        final_row = []
        for i in range(self.columns):
            final_row.append(f' {i+1} ')
        result.append("|" + "|".join(final_row) + "|")
        return "\n".join(result) + "\n"

    def __str__(self) -> str:
        return self.highlighted_str()

    #endregion

    def insert(self, piece, col) -> int:
        if type(piece) is not Piece or piece is Piece.EMPTY:
            raise InvalidPieceError(f'Invlid piece {repr(piece)}')
        if col not in range(self.columns):
            raise OutOfBoundsError(
                f'This column ({one_index(col)}) is out of bounds.')

        r = self.first_available_row(col)
        if r == -1:
            raise InvalidInsertError(
                f"This column ({one_index(col)}) is full.")

        self.board[r][col] = piece
        self.next_player = self.opponent(piece)
        if not self.pretend:
            self.log.normal(
                f'{repr(piece)}{" (pretend)" if self.pretend else ""} played a piece in column {one_index(col)}.'
            )
        return r

    def available_columns(self) -> list[int]:
        return [
            c for c in range(self.columns) if self.board[0][c] is Piece.EMPTY
        ]

    def first_available_row(self, col) -> int:
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] is Piece.EMPTY:
                return r
        return -1

    def move_scores(self, piece) -> list[int]:
        """For each possible move (column), evaluate all relevant groups of four. Return a score for each column representing the aggregate benefit of putting a piece in that column."""
        _INTERVAL = 100
        other = self.opponent(piece)
        scores = [0] * self.columns
        for c in range(self.columns):
            groups = []
            r = self.first_available_row(c)
            # if row is full, skip it.
            if r == -1:
                continue
            # These are the start points from which to collect valid groups of 4.
            for off in range(3, -1, -1):
                # Vertical groups of 4 that include r, starting at top.
                startR = r - off
                safe_r = startR >= 0 and startR + 3 < self.rows
                if safe_r:
                    groups.append([(startR + i, c) for i in range(4)])

                # Horizontal groups of 4 that include r, starting at left.
                startC = c - off
                safe_c = startC >= 0 and startC + 3 < self.columns
                if safe_c:
                    groups.append([(r, startC + i) for i in range(4)])

                # \ diagonals groups of 4 that include r, starting at top left.
                if safe_r and safe_c:
                    groups.append([(startR + i, startC + i) for i in range(4)])

                # / diagonals groups of 4 that include r, starting at bottom left.
                startR = r + off
                safe_r = startR < self.rows and startR - 3 >= 0
                if safe_r and safe_c:
                    groups.append([(startR - i, startC + i) for i in range(4)])
            # The score for each group of 4 positions that contains position (r,c) contributes to the "score" for this column.
            for window in groups:
                scores[c] += self.window_score(window, piece)
            # Favor plays in emptier rows.
            row_modifier = r * 15
            scores[c] += row_modifier
            # Add to score if number of neighboring opponent pieces is high.
            neighboring_opponents = 0
            for i in range(-1, 2, 2):
              safe_r = r+i in range(self.rows)
              safe_c = c+i in range(self.columns)
              if safe_r and self.board[r+i][c] == other:
                neighboring_opponents += 1
              if safe_c and self.board[r][c+i] == other:
                neighboring_opponents += 1
              if safe_r and safe_c and self.board[r+i][c+i] == other:
                neighboring_opponents += 1
            scores[c] += neighboring_opponents * 5
            # scores[c] = _INTERVAL * round(scores[c]/_INTERVAL)
        return scores
    
    def window_score(self, positions, piece=Piece.RED) -> int:
        """Returns a score indicating how advantageous the move would be."""
        _WIN = 1000
        score = 0
        if len(positions) != 4:
            self.log.debug(f'window_score: warning - not a four element list: {positions}')
        pieces = [self.board[r][c] for r, c in positions]
        is_diagonal = positions[0][0] is not positions[1][0] and positions[0][1] is not positions[1][1]
        low_row = ConnectFour.lowest_row(positions)
        other = self.opponent(piece)
        mine, yours = pieces.count(piece), pieces.count(other)
        blanks = len(positions) - mine - yours
        if mine == 4 or yours == 4:  # Someone has already won with this group.
            score = _WIN
        elif mine == 3 and blanks == 1:  # This is potentially a winning group for "piece".
            score = 800
        elif yours == 3 and blanks == 1:  # This is potentially a winning group for "other".
            score = 700
        elif mine > 0 and yours == 0:  # There is potential for "piece".
            score = (mine * 15)
        elif yours > 0 and mine == 0: # There is potential for "other".
            score = (yours * 15)
        else: # Neither can win this group.
            score = 0
        return score

    def get_best_move(self, piece) -> int:
        """Return the best move (column) based on the scores for each column and predicting a few moves into the future."""
        _SCORE_GRANULARITY = 50
        col = -1
        other = self.opponent(piece)
        available = self.available_columns()
        if len(available) > 0:
            self.log.debug("available -", available)

            # Evaluate the score of each available move.
            scores = self.move_scores(piece)
            self.log.debug("scores -", scores)
            for c in available:
                # If this is the real game, try to predict the future.
                if not self.pretend:
                    # Set up a fake environment to predict the results of the move.
                    prediction = ConnectFour(board=deepcopy(self.board),
                                             pretend=True)
                    prediction.insert(piece, c)

                    # Play a few turns to see if we can set up a win or prevent a loss.
                    # Update the calculated score, if so.
                    turns = 5
                    for turn in range(1, turns+1):
                        predicted_winner = prediction.play(1) and prediction.board_winner()
                        if predicted_winner in [Piece.RED, Piece.BLACK]:
                            # The current advantage we get from playing here depends on whether or not the predicted winner is us.
                            # Subtract the points if the other player is predicted to win.
                            sign = 1 if predicted_winner == piece else -1
                            prediction_adjustment = sign * round((turns+1) * 100 / turn)
                            scores[c] += prediction_adjustment
                            self.log.debug("predicted for", c, repr(predicted_winner), f"in {turn} turns", prediction_adjustment)
                            break
                scores[c] = _SCORE_GRANULARITY * round(scores[c] / _SCORE_GRANULARITY)
            # Determine best move from scores for all possibilities
            self.log.debug("scores -", scores)
            valid_scores = [s for i, s in enumerate(scores) if i in available]
            best_score = max(valid_scores)
            best_cols = [i for i, s in enumerate(scores) if s == best_score and i in available]
            col = random.choice(best_cols)
        self.log.verbose(f'get_best_move({repr(piece)}):',
                         f'column {one_index(col)}')
        return col

    #region Game Completion
    def board_full(self) -> bool:
        return len(self.available_columns()) == 0

    def board_winner(self) -> Piece:
        groups = []
        for c in range(0, self.columns):
            # Whether or not there is sufficient room below the current c to find a group of four.
            space_down = c < self.columns - 3
            for r in range(0, self.rows):
                # for down to up diagonals, we'll mirror r on the horizontal midpoint.
                opposite_r = self.rows - r - 1
                # Whether or not there is sufficient room to the right or left of the current r to find a group of four.
                space_right = r < self.rows - 3
                space_left = opposite_r > 2

                # horizontal group that includes r.
                if space_right:
                    groups.append([(r + j, c) for j in range(4)])
                # vertical group that includes r.
                if space_down:
                    groups.append([(r, c + j) for j in range(4)])
                # up to down diagonal group that includes r.
                if space_down and space_right:
                    groups.append([(r + j, c + j) for j in range(4)])
                # down to up diagonal group that includes opposite_r.
                if space_down and space_left:
                    groups.append([(opposite_r - j, c + j) for j in range(4)])

        for window in groups:
            if self.window_score(positions=window) == 1000:
                r, c = window[0]
                self.winner = self.board[r][c]
                self.winning_group = window
                self.log.debug("board_winner:", repr(self.winner), window)
                return self.winner
        return Piece.EMPTY

    def board_has_winner(self):
        return self.board_winner() != Piece.EMPTY

    def game_over(self):
        return self.board_full() or self.board_has_winner()

    #endregion

    #region Turn Mechanics
    def play(self, max_turns=42) -> bool:
        turns = 0
        if self.players == 1:
            human_piece = self.get_color_choice()
        row, col = -1, -1
        while not self.game_over() and turns < max_turns:
            if not self.pretend:
                self.log.normal(self.highlighted_str([(row, col)]))
            is_human_turn = (self.players == 1 and human_piece is
                             self.next_player) or self.players == 2
            if is_human_turn:
                col = self.get_column_choice(self.next_player)
            else:
                col = self.get_best_move(self.next_player)

            try:
                row = self.insert(self.next_player, col)
                turns += 1
            except Exception as e:
                self.log.normal(self)
                self.log.error(e)
                self.log.normal("\nPlease try again.\n")
                return False

        if not self.pretend:
            highlight = self.winning_group if len(
                self.winning_group) > 0 else [(row, col)]
            self.log.normal(self.highlighted_str(highlight))
            self.log.normal("Game finished!")
            self.log.normal(f'Winner: {repr(self.winner)}')
        return True

    def get_color_choice(self) -> Piece:
        request, color = 0, ""
        while color.upper() not in [Piece.BLACK.name, Piece.RED.name]:
            if request > 0:
                self.log.normal("Invalid selection.")
            color = input("Which color would you like to use: RED or BLACK?\n")
            request += 1
            if request > 5:
                raise Exception("Too many failed attempts")
        return Piece[color.upper()]

    def get_column_choice(self, piece) -> int:
        col, request = 0, 0
        while col not in range(1, self.columns + 1):
            col = input(
                f'{repr(piece)}, please select a column number (1-{self.columns}):\n'
            )
            try:
                col = int(col)
                request += 1
            except:
                self.log.normal("Invalid column.")
                if request > 5:
                    raise Exception("Too many failed attempts")
        return col - 1

    def get_number_players(self) -> int:
        players = -1
        request = 0
        max_attempts = 5
        while players not in [0, 1, 2] and request < max_attempts:
            if request > 0:
                self.log.normal(f'Invalid selection. {request} of 5 attempts.')
            players = input(
                "How many human players would like to participate?\n")
            try:
                players = int(players)
                if players not in [0, 1, 2]:
                    raise InvalidPlayersError(
                        f"{players} is not a valid option.")
            except Exception as e:
                self.log.error(e)
            request += 1
        self.log.info(f'{players} human players.')
        return players

    #endregion


def main() -> None:
    game = ConnectFour.new(players=0, log_level=LogLevel.DEBUG)
    # game = ConnectFour.new()
    game.play()


if __name__ == "__main__":
    main()
