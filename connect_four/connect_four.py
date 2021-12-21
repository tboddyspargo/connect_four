from enum import Enum, IntEnum, auto
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, Optional
from sys import intern
from traceback import format_exc
from copy import deepcopy
import random
import time
import functools

#region Globals
WINNING_SCORE = 1000
POINTS_PER_PIECE = 2
OPPONENT_SCORE_MODIFIER = 1.5
#endregion

#region Helpers
def one_index(n: int) -> int:
    if not n is int:
        return n
    return n + 1
#endregion

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
        print(format_exc())

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

#region Timing
# Credit for this timer class that behaves like a context manager and decorator goes to Real Python (https://realpython.com/python-timer/).
@dataclass
class Timer:
    timers: ClassVar[Dict[str, float]] = dict()
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = Logger(level = LogLevel.DEBUG).debug
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time
    
    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()

    def __call__(self, func):
        """Support using Timer as a decorator"""
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper_timer
#endregion

#region Error Handling
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

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

#endregion

#region Board Components
class Piece(Enum):
    EMPTY = intern(" ")
    RED = intern("X")
    BLACK = intern("O")

    def __str__(self) -> str:
        return f'{self.name} ({self.value})'

    def __repr__(self) -> str:
        return self.value

# TODO: add class for each slot/position?
#endregion
class ConnectFour:
    #region Attributes and Printing
    rows: int = 6
    columns: int = 7
    slots: list[tuple[int]] = []
    slots_by_position: dict[tuple[int], list] = {}
    def __init__(self,
                 players: int = 0,
                 human_player: Piece = None,
                 board: list[list[Piece]] = None,
                 pretend: bool = False,
                 log_level: LogLevel = LogLevel.NONE,
                 rows: int = 6,
                 columns: int = 7) -> None:
        self.log: Logger = Logger(log_level)
        self.pretend: bool = pretend
        self.players: int = players
        self.human_player: Piece = self.prompt_for_color_choice() if human_player is None and self.players == 1 else human_player
        self.rows: int = rows
        self.columns: int = columns
        self.board: list[list[Piece]] = board or [[Piece.EMPTY for c in range(self.columns)] for r in range(self.rows)]
        self.starting_player: Piece = Piece.RED
        self.current_turn: int = self.determine_turn()
        self.winning_group: list[tuple] = []
        self.winner: Piece = Piece.EMPTY
        self.slots: list[tuple[int]] = self.get_slots()
        self.slots_by_position: dict[tuple[int], list[tuple[int]]] = self.get_slots_by_position()
        self.minimax_results: dict[str, int] = {}
        self.plays: list[tuple[int]] = []

    @classmethod
    def new(cls, **kwargs) -> 'ConnectFour':
        instance = cls(**kwargs)
        if 'players' not in kwargs:
            instance.players = instance.prompt_for_number_players()
        return instance

    @staticmethod
    def lowest_row(positions) -> int:
        """Return the lowest row in the given set of board positions."""
        result = -1
        for r,c in positions:
            if r > result:
                result = r
        return result

    @staticmethod
    def opponent(player: Piece) -> Piece:
        """Returns the opponent of the given piece. There are only two piece choices, so this will be obvious. If None is provided, None will be returned."""
        if player is Piece.BLACK:
            return Piece.RED
        elif player is Piece.RED:
            return Piece.BLACK
        return None

    def determine_turn(self) -> int:
        """Return the current turn number based on how many positions on the board are already occupied."""
        occupied_positions = 0
        for c in range(self.columns):
            for r in range(self.rows-1, -1, -1):
                if self.board[r][c] is Piece.EMPTY:
                    break
                occupied_positions += 1
        return occupied_positions + 1
    
    def player_whose_turn_it_is(self) -> Piece:
        """Return the Piece of the player whose turn it is based on the starting player and the current turn number."""
        return self.starting_player if self.current_turn % 2 == 1 else self.opponent(self.starting_player)

    def highlighted_str(self, highlight_positions: list[Piece] = []):
        """Return a string representation of the board with an optional set of positions highlighted."""
        row_div = "|---------------------------|"
        result = []
        for r in range(self.rows):
            row = []
            for c in range(self.columns):
                piece_str = f" {self.board[r][c].value} "
                if (r, c) in highlight_positions:
                    piece_str = "|" + piece_str[1] + "|"
                row.append(piece_str)
            result.append("|" + "|".join(row) + "|")

        result.append(row_div)
        final_row = []
        for i in range(self.columns):
            final_row.append(f" {i+1} ")
        result.append("|" + "|".join(final_row) + "|")
        return "\n".join(result) + "\n"

    @classmethod
    def get_shape_from_positions(cls, positions: list[tuple[int]]) -> str:
        change_r = positions[1][0] - positions[0][0]
        change_c = positions[1][1] - positions[0][1]
        shape = "–" if change_r == 0 else "|"
        if change_r != 0 != change_c:
            shape = "/" if change_r < 0 else "\\"
        return shape
    

    @classmethod
    def get_slots(cls):
        results = []
        for c in range(cls.columns):
            # Whether or not there is sufficient room to the right of c to find a group of four.
            space_right = c < cls.columns - 3
            for r in range(cls.rows):
                # for down to up diagonals, we'll mirror r on the vertical midpoint.
                opposite_r = cls.rows - r - 1
                # Whether or not there is sufficient room below and above the current r to find a group of four.
                space_down = r < cls.rows - 3
                space_up = opposite_r > 2

                # | group that includes c.
                if space_down:
                    results.append([(r + j, c) for j in range(4)])
                # – group that includes r.
                if space_right:
                    results.append([(r, c + j) for j in range(4)])
                # \ diagonal group that includes r.
                if space_down and space_right:
                    results.append([(r + j, c + j) for j in range(4)])
                # / diagonal group that includes opposite_r.
                if space_up and space_right:
                    results.append([(opposite_r - j, c + j) for j in range(4)])
        return results

    @classmethod
    def get_slots_by_position(cls):
        result = {}
        cls.slots = cls.slots or cls.get_slots()
        for c in range(cls.columns):
            for r in range(cls.columns):
                p = (r,c)
                result[p] = []
                for s in cls.slots:
                    if p in s:
                        result[p].append([new_p for new_p in s if new_p[0] != p[0] or new_p[1] != p[1]])
        return result

    def __str__(self) -> str:
        """Return a basic string representation of the board without any highlighted positions."""
        return self.highlighted_str()

    def __repr__(self) -> str:
        """Return a string representing every position and piece on the board."""
        result = []
        for r in range(self.rows):
            for c in range(self.columns):
                if self.board[r][c] is not Piece.EMPTY:
                    result.append(f'{r}{c}{self.board[r][c].value}')
        return ",".join(result)

    #endregion

    #region Column and Row Interactions
    def insert(self, col: int, player: Piece = None) -> int:
        if col not in range(self.columns):
            raise OutOfBoundsError(f'insert(): column {one_index(col)} is out of bounds.')
        row = self.first_available_row(col)
        if row == -1:
            raise InvalidInsertError(f"insert(): column {one_index(col)} is full.")
        if player is None:
            player = self.player_whose_turn_it_is()

        self.board[row][col] = player
        self.current_turn += 1
        self.plays.append((row, col))
        if not self.pretend:
            self.log.normal(f'{self.board[row][col]} played a piece in column {one_index(col)}.')
        return row
    
    def remove(self, col: int) -> None:
        if col not in range(self.columns):
            raise OutOfBoundsError(f'remove(): column {one_index(col)} is out of bounds.')
        
        for row in range(self.rows):
            if self.board[row][col] is not Piece.EMPTY:
                old_piece = self.board[row][col]
                self.board[row][col] = Piece.EMPTY
                self.current_turn -= 1
                if (row, col) not in self.plays:
                    self.plays.remove((row, col))
                self.evaluate_board_win()
                if not self.pretend:
                    self.log.error(f'{old_piece} removed their piece from column {one_index(col)}; Current turn is now {self.current_turn}')
                return
        raise InvalidRemoveError(f"remove(): there are no pieces in column {one_index(col)} to remove.")

    @classmethod
    def prioritize_central_columns(cls, col):
        mid = cls.columns / 2
        return round(abs((mid-0.5) - col))

    def available_columns(self) -> list[int]:
        available = [c for c in range(self.columns) if self.board[0][c] is Piece.EMPTY]
        available.sort(key = self.prioritize_central_columns)
        return available

    def first_available_row(self, col: int) -> int:
        if col not in range(self.columns):
            raise OutOfBoundsError(f'Column {one_index(col)} is out of bounds.')
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] is Piece.EMPTY:
                return r
        return -1
    #endregion

    def new_window_score(self, positions: list[tuple[int, int]]) -> int:
        pieces = [self.board[r][c] for r,c in positions]
        # Count occurrences
        totals = { Piece.RED: 0, Piece.BLACK: 0, Piece.EMPTY: 0 }
        for p in pieces:
            totals[p] += 1
        # Compute and return points based on totals.
        if totals[Piece.BLACK] == 0:
            if totals[Piece.RED] == 4:
                return WINNING_SCORE
            return (POINTS_PER_PIECE * (totals[Piece.RED]**2))
        elif totals[Piece.RED] == 0:
            if totals[Piece.BLACK] == 4:
                return -WINNING_SCORE
            return -(POINTS_PER_PIECE * (totals[Piece.BLACK]**2))
        return 0

    def new_move_scores(self, player: Piece = None, available: list[tuple[int]] = None) -> list[int]:
        """Returns a board score for every column reflecting how advantageous it is for either player.
        The score will be positive if it favors Piece.RED and negative if it favors Piece.BLACK or 0 if there's no obvious benefit to either.
        """
        if available is None:
            available = self.available_columns()
        if player is None:
            player = self.player_whose_turn_it_is()
        opponent = self.opponent(player)
        game = ConnectFour(board = deepcopy(self.board), players = 0, pretend = True, log_level = self.log.level, rows = self.rows, columns = self.columns)
        scores = [0] * self.columns
        opponent_scores = [0] * self.columns
        for c in available:
            game.insert(c, player)
            scores[c] = game.evaluate_board_opportunities(player)
            game.remove(c)
            game.insert(c, opponent)
            opponent_scores[c] = game.evaluate_board_opportunities(opponent)
            game.remove(c)
        for c in available:
            scores[c] -= int(opponent_scores[c] / OPPONENT_SCORE_MODIFIER)
        return scores

    def evaluate_board_win(self) -> int:
        for s in self.slots:
            score = self.new_window_score(s)
            if WINNING_SCORE == score or -WINNING_SCORE == score:
                self.winning_group = s
                self.winner = self.board[s[0][0]][s[0][1]]
                return score
        return 0
    
    def new_board_winner(self, win: int = None) -> Piece:
        if win is None:
            win = self.evaluate_board_win()
        if win == WINNING_SCORE:
            return Piece.RED
        elif win == -WINNING_SCORE:
            return Piece.BLACK
        return None

    def evaluate_board_opportunities(self, player: Piece = None) -> int:
        result = 0
        for p, slots in self.slots_by_position.items():
            for s in slots:
                result += self.new_window_score([p] + s)
        return result

    def move_scores(self, player: Piece = None) -> list[int]:
        """For each possible move (column), evaluate all relevant groups of four.
        Return a score for each column representing the aggregate benefit of putting a piece in that column."""
        # TODO: Improve performance at the cost of space by using a dictionary to store already calculated window scores that include a given position.
        if player is None:
            player = self.player_whose_turn_it_is()
        opponent = self.opponent(player)
        scores = [0] * self.columns
        for c in range(self.columns):
            groups = []
            r = self.first_available_row(c)
            # If row is full, skip it.
            if r == -1:
                continue
            for w in self.get_slots_by_position[(r,c)]:
                # The score for each group of 4 positions that contains position (r,c) contributes to the "score" for this column.
                scores[c] += self.window_score([(r,c)] + w, target_player = player, target_position = (r,c))

            # TODO: Determine if necessary.
            # Favor plays in emptier rows.
            row_modifier = r * 5
            scores[c] += row_modifier
            # Add to score if number of neighboring opponent pieces is high.
            neighboring_opponents = 0
            for i in range(r-1, r+2, 1):
                for j in range(c-1, c+2, 1):
                    if not (i == r and j == c) and i in range(self.rows) and j in range(self.columns) and self.board[i][j] == opponent:
                        neighboring_opponents += 1
            scores[c] += neighboring_opponents * 2
        return scores

    def window_score(self, positions: list[tuple[int, int]], target_player: Piece = None, target_position: tuple = None) -> int:
        """Returns a score indicating how advantageous the move would be."""
        _WIN: int = 1000
        score: int = 0
        if len(positions) != 4:
            self.log.debug(f'window_score: warning - not a four element list: {positions}')
        
        opponent: Piece = self.opponent(target_player)
        target_index: int = -1 if target_position is None else positions.index(target_position)

        # Convert the positions to a list of pieces.
        pieces: list[Piece] = [self.board[r][c] for r, c in positions]

        # Count up the total and consecutive pieces in these positions.
        last_piece: Piece = Piece.EMPTY
        total: dict[str, int] = {
            Piece.RED: 0,
            Piece.BLACK: 0,
            Piece.EMPTY: 0
        }
        consecutive: dict[str, tuple[int]] = {
            Piece.RED: (-1, -1),
            Piece.BLACK: (-1, -1)
        }
        for i, p in enumerate(pieces):
            # Count totals
            total[p] += 1
            
            # Count consecutive occurrences, consider the target position as a continuation if it occurs after a non-empty position.
            if (p is not Piece.EMPTY and p is last_piece) or (last_piece is not Piece.EMPTY and i is target_index < 3):
                consecutive[last_piece] = (consecutive[last_piece][0], i)
            elif p is not Piece.EMPTY and consecutive[p][1] - consecutive[p][0] < 1:
                consecutive[p] = (i, i)
            if i is not target_index:
                last_piece = p

        if total[Piece.RED] == 4 or total[Piece.BLACK] == 4: # Someone has already won with this group.
            score = _WIN
        elif None not in {target_player, opponent}: # We're evaluating a potential move (as opposed to simply reading what's on the board).
            # What is the orientation of these positions on the board?
            is_diagonal: bool = positions[0][0] is not positions[1][0] and positions[0][1] is not positions[1][1]
            is_horizontal: bool = positions[0][0] is positions[1][0]

            # Determine the contents of the positions just before and after this series.
            first, last = positions[0], positions[3]
            change_r, change_c = (last[0] - first[0]) // 3, (last[1] - first[1]) // 3
            before: Piece = None
            after: Piece = None
            if (before_r := first[0] - change_r) in range(self.rows) and (before_c := first[1] - change_c) in range(self.columns):
                before = self.board[before_r][before_c]
            if (after_r := last[0] + change_r) in range(self.rows) and (after_c := last[1] + change_c)  in range(self.columns):
                after = self.board[after_r][after_c]
            
            # Determine whether this move could create two overlapping win situations.
            double_possibility = False
            even_odd_advantage = False
            if is_horizontal:
                even_odd_player = (Piece.RED, Piece.BLACK)[target_position[0] % 2]
                even_odd_advantage = target_player == even_odd_player

                if total[Piece.EMPTY] == 2 and any([consecutive_piece := Piece(k) for k, v in consecutive.items() if v[1] - v[0] + 1 >= 2]):
                    consecutive_positions = consecutive[consecutive_piece]
                    
                    # If there's a possible 3-in-a-row in these positions and an adjacent blank just outside of these positions, that's a double possibility.
                    earlier_possibility = target_index == 0 and min(consecutive_positions) == 1 and before in {consecutive_piece, Piece.EMPTY}
                    later_possibility = target_index in {2, 3} and max(consecutive_positions) >= 2 and after in {consecutive_piece, Piece.EMPTY}
                    double_possibility = (earlier_possibility or later_possibility)
            


            # Weight strategic moves higher.
            double_modifier = 10 * ((target_position[0]+1)//2) if double_possibility else 0
            even_odd_modifier = 5 if even_odd_advantage else 0
            mod = (double_modifier + even_odd_advantage) or 1

            # Score the potential advantageousness of this move.
            if total[Piece.EMPTY] == 1 and 3 in {total[target_player], total[opponent]}: # Three matching + one empty.
                score = round((total[target_player] * 300 ) + (total[opponent] * 266), -2) # round to nearest hundred
            elif total[Piece.EMPTY] == 2 and 2 in {total[target_player], total[opponent]}: # Two matching + two empty.
                score = (total[target_player] * (10 + mod)) + (total[opponent] * (15 + mod))
            elif total[Piece.EMPTY] == 3 and 1 in {total[target_player], total[opponent]}: # One player + three empty.
                score = (total[target_player] * (0 + mod)) + (total[opponent] * (5 + mod))
            else: # Some amount of both players.
                score = 0
        
        # Blanks are helpful, no matter where they are.
        if score < _WIN/2:
            score +=  total[Piece.EMPTY] * 5

        # Log some debug info.
        if not self.pretend and target_position is not None and score > 30:
            shape = self.get_shape_from_positions(positions)
            self.log.debug(one_index(target_position[1]), one_index(positions[0][1]), shape, score, mod, pieces)
        return score

    def get_best_current_move(self) -> int:
        """Return the best move (column) based on the scores for each column and predicting a few moves into the future."""
        col = -1
        player = self.player_whose_turn_it_is()
        opponent = self.opponent(player)
        available = self.available_columns()
        if len(available) > 0:
            self.log.debug("available -", available)

            # Evaluate the score of each available move.
            scores = self.move_scores(player)
            self.log.debug("scores -", scores)
            best_score = max(scores)
            if best_score < 700 and len(available) > 1 and not self.pretend:
                # If this is the real game and there aren't any inherently obvious moves (i.e. not winning or blocking),
                # try to predict the future.
                for c in available:
                    # Set up a fake environment to predict the results of the move.
                    prediction = ConnectFour(board=deepcopy(self.board),
                                            pretend=True)
                    prediction.insert(c)
                    # Play a few turns to see if we can set up a win or prevent a loss.
                    # Update the calculated score, if so.
                    turns = 5
                    for turn in range(1, turns+1):
                        prediction.play(1)
                        # If the current player would lose after this turn, we should avoid this play.
                        # Only adjusting for predicted losses allows us to favor both winning and blocking plays.
                        if (predicted_winner := prediction.get_board_winner()) is not Piece.EMPTY:
                            multiplier = -100 if predicted_winner is opponent else 25
                            prediction_adjustment = round((turns+1) / turn) * multiplier
                            scores[c] += prediction_adjustment
                            if turn < 3:
                                self.log.debug(f"If {player} plays in column {one_index(c)}, {predicted_winner} is predicted to WIN in {turn} turns", prediction_adjustment)
                            break
            # Determine best move from scores for all possibilities
            self.log.debug("scores -", scores)
            valid_scores = [s for i, s in enumerate(scores) if i in available]
            best_score = max(valid_scores)
            best_cols = [i for i, s in enumerate(scores) if s == best_score and i in available]
            col = random.choice(best_cols)
        self.log.verbose(f'get_best_move({player}):', f'column {one_index(col)}')
        return col

    def minimaxAll(self, player: Piece = None, depth: int = 4) -> list[int]:
        if player is None:
            player = self.player_whose_turn_it_is()
        game = self
        if not self.pretend:
            game = ConnectFour(players = 0, board = deepcopy(self.board), pretend = True, log_level = self.log.level, rows = self.rows, columns = self.columns)
        self.log.verbose("minimaxAll:", repr(player), depth)
        scores = [0] * self.columns
        for col in self.available_columns():
            self.log.verbose("\t"*(5-depth), f"minimaxAll: {repr(player)} {depth} playing in col {col} on turn {self.current_turn}")
            game.insert(col)
            scores[col] = game.minimax(player = game.opponent(player), depth = depth)
            game.remove(col)
            self.log.verbose("\t"*(5-depth), f"minimaxAll: {repr(player)} {depth} playing in col {col} would result in {scores[col]}")
        return scores

    def minimax(self, player: Piece = None, depth: int = 4, alpha: int = -1000, beta: int = 1000) -> int:
        string_repr = repr(self)
        if player is None:
            player = self.player_whose_turn_it_is()
        # self.log.verbose("minimax:", repr(player), depth, "|", alpha, beta)
        is_max = player is Piece.RED
        sign = 1 if is_max else -1
        score = 0
        # If the board outcome has been determined or the intended depth has been reached, return.
        # Otherwise, make recursive call to go deeper.
        if string_repr in self.minimax_results:
            score = self.minimax_results[string_repr]
            self.log.verbose("\tminimax: got value from memo:", score, string_repr)
        win = self.evaluate_board_win()
        winner = self.new_board_winner(win)
        if winner is not None:
            # If there is already a winner, then the win belongs to the opponent.
            self.minimax_results[string_repr] = win
            win_sign = 1 if win > 0 else -1
            score = win - (win_sign * self.current_turn)
            self.log.verbose("\t"*(5-depth), f"'{repr(self.winner)}' win:", win, win_sign, score)
        elif self.board_full():
            score = 0
        elif depth < 1:
            # Since this isn't a true stalemate guarantee, don't save this in minimax_results.
            score = self.evaluate_board_opportunities(player = player)
        else:
            best = sign * -10000000
            for col in self.available_columns():
                self.log.verbose("\t"*depth, f"{repr(player)} {depth} playing in col {col} on turn {self.current_turn}, alpha: {alpha}, beta: {beta}")
                self.insert(col)
                val = self.minimax(player = self.opponent(player), depth = depth - 1, alpha = alpha, beta = beta)
                self.remove(col)
                
                if (is_max and val > best) or (not is_max and val < best):
                    best = val

                if is_max and best > alpha:
                    alpha = best
                elif not is_max and best < beta:
                    beta = best
                
                if beta <= alpha:
                    break
                self.log.verbose("\t"*(5-depth), f"{repr(player)} {depth} playing in col {col} would result in {score} (best: {best}), alpha: {alpha} beta: {beta}")
            score = best
        
        return score

    @Timer(text = "get_best_move: took {:.1f}s")
    def get_best_move(self) -> int:
        if self.board_full():
            raise BoardFullError("get_best_move: cannot evaluate the best move of a full board.")
        player = self.player_whose_turn_it_is()
        is_max = player is Piece.RED
        self.log.debug("get_best_move:", repr(player), "MAX" if is_max else "MIN")
        min_or_max_func = max if is_max else min
        best_columns = self.available_columns()

        # use optimal moves based only on current board state and the results of minimax.
        simple_scores = self.new_move_scores(player = player, available = best_columns)
        self.log.debug(f"\tsimple optimum {repr(player)} scores:", simple_scores)
        weights = [simple_scores[c] if c in best_columns else 0 for c in range(self.columns)]
        winning_or_blocking_columns = [c for c, s in enumerate(weights) if s < -WINNING_SCORE or s > WINNING_SCORE]
        if any(winning_or_blocking_columns):
            best_columns = self.best_columns_from_scores(weights, min_or_max_func, winning_or_blocking_columns)
            self.log.debug("\tobvious moves:", best_columns, "best:", best_columns[0])
            return best_columns[0]

        # Get the minimax score for every possible move.
        scores = self.minimaxAll(player = player, depth = 3)
        best_columns = self.best_columns_from_scores(scores, min_or_max_func)
        best_score = min_or_max_func([s for c, s in enumerate(scores) if c in best_columns])
        self.log.debug("\tminmax scores:", scores, best_columns, "best:", best_score)
        best_columns = [c for c in best_columns if scores[c] == best_score]
        self.log.debug("\tminimax moves:", best_columns)
        if len(best_columns) == 1:
            self.log.debug("\tbest minimax move:", best_columns[0])
            return best_columns[0]

        # If multiple moves have the same minimax score, weight them by 
        best_columns = self.best_columns_from_scores(weights, min_or_max_func, best_columns)
        self.log.debug("\tsimple optimum columns:", weights)
        if len(best_columns) == 1:
            self.log.debug("\tcombined minimax and optimal move:", best_columns[0])
            return best_columns[0]
        
        best_score = min_or_max_func([s for c, s in enumerate(weights) if c in best_columns])
        random_choice = random.choice([c for c, s in enumerate(weights) if best_score == s and c in best_columns])
        self.log.debug("\trandom central minimax move:", random_choice)
        return random_choice
            

            
    def best_columns_from_scores(self, scores: list[int], best_func: Callable[[list[int]], int] = max, available: list[int] = None) -> int:
        if available is None:
            available = self.available_columns()
        self.log.verbose("\tbest_columns_from_score:", scores, available)
        best_score = best_func([s for c, s in enumerate(scores) if c in available])
        best_available = [c for c, s in enumerate(scores) if s == best_score and c in available]
        best_available.sort(key = self.prioritize_central_columns)
        self.log.verbose("\tbest_columns:", best_available)
        return best_available


    #region Game Completion
    def board_full(self) -> bool:
        return len(self.available_columns()) == 0

    def get_board_winner(self) -> Piece:
        win = []
        winner = Piece.EMPTY
        for c in range(self.columns):
            # Whether or not there is sufficient room to the right of c to find a group of four.
            space_right = c < self.columns - 3
            for r in range(self.rows):
                # for down to up diagonals, we'll mirror r on the vertical midpoint.
                opposite_r = self.rows - r - 1
                # Whether or not there is sufficient room below and above the current r to find a group of four.
                space_down = r < self.rows - 3
                space_up = opposite_r > 2

                # | group that includes c.
                if space_down and self.window_score(window := [(r + j, c) for j in range(4)]) == 1000:
                    win = window
                    break
                # – group that includes r.
                if space_right and self.window_score(window := [(r, c + j) for j in range(4)]) == 1000:
                    win = window
                    break
                # \ diagonal group that includes r.
                if space_down and space_right and self.window_score(window := [(r + j, c + j) for j in range(4)]) == 1000:
                    win = window
                    break
                # / diagonal group that includes opposite_r.
                if space_up and space_right and self.window_score(window := [(opposite_r - j, c + j) for j in range(4)]) == 1000:
                    win = window
                    break

        if win:
            r, c = win[0]
            winner = self.board[r][c]
            self.log.debug("board_winner:", winner, self.get_shape_from_positions(win), win)
        self.winner = winner
        self.winning_group = win
        return self.winner

    def board_has_winner(self):
        return self.new_board_winner() in {Piece.RED, Piece.BLACK}

    def game_over(self):
        return self.board_full() or self.board_has_winner()

    #endregion

    #region Turn Mechanics
    def play(self, turns = None) -> bool:
        max_turns = self.rows * self.columns if turns is None else min(self.rows * self.columns, self.current_turn - 1 + turns)
        human_players: list[Piece] = [Piece.RED, Piece.BLACK] if self.players == 2 else [self.human_player] * self.players

        # Print the board if this is a real game.
        if not self.pretend:
            self.log.normal(self)
        row, col, attempt = -1, -1, 1
        while not self.game_over() and self.current_turn <= max_turns:

            # Get the player's move choice (which column they'll put a piece into).
            if self.player_whose_turn_it_is() in human_players:
                col = self.prompt_for_column_choice()
            else:
                col = self.get_best_move()

            # Attempt the move
            try:
                row = self.insert(col)
                attempt = 1
                # Print the board if this is a real game.
                if not self.pretend:
                    self.log.normal(self.highlighted_str([(row, col)]))
            except Exception as e:
                self.log.error(e)
                if self.player_whose_turn_it_is() in human_players:
                    self.log.normal("Please try again.")
                    attempt += 1
                else:
                    return False
        if turns is not None:
            self.log.debug(f"play: finished playing {turns} turns.")
        # If this is a real game and the game has ended, print the results.
        if not self.pretend and (self.board_full() or self.winner != Piece.EMPTY):
            highlight = self.winning_group if len(self.winning_group) > 0 else [(row, col)]
            self.log.normal(self.highlighted_str(highlight))
            self.log.normal("Game finished!")
            if self.winner in [Piece.RED, Piece.BLACK]:
                self.log.normal('Winner:', self.winner)
            else:
                self.log.normal('Stalemate')
        return True

    def prompt_for_color_choice(self) -> Piece:
        request, color = 0, ""
        while color.upper() not in [Piece.BLACK.name, Piece.RED.name]:
            if request > 0:
                self.log.normal("Invalid selection.")
            color = input(f"Which color would you like to use: {Piece.RED} or {Piece.BLACK}?\n")
            request += 1
            if request > 5:
                raise Exception("Too many failed attempts")
        return Piece[color.upper()]

    def prompt_for_column_choice(self) -> int:
        col, request = -1, 0
        while col not in range(self.columns):
            col = input(f'{self.player_whose_turn_it_is()}, please select a column number (1-{one_index(self.columns)}):\n')
            try:
                col = int(col) - 1
                request += 1
            except:
                self.log.normal("Invalid column.")
                if request > 5:
                    raise Exception("Too many failed attempts")
        return col

    def prompt_for_number_players(self) -> int:
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
                    raise InvalidPlayersError(f"{players} is not a valid option.")
            except Exception as e:
                self.log.error(e)
            request += 1
        self.log.info(f'{players} human players.')
        return players

    #endregion

@Timer(text = "main took {:.1f}s")
def main() -> None:
    game = ConnectFour.new(players=0, human_player = Piece.RED, log_level=LogLevel.DEBUG)
    # game = ConnectFour.new()

    # game.play(6)
    game.play()

if __name__ == "__main__":
    main()
