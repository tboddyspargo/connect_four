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


class InvalidPlayersError(Error):
    """Invalid number of players provided."""
    pass


#endregion


class Piece(Enum):
    EMPTY = " "
    RED = "X"
    BLACK = "O"

    def __str__(self) -> str:
        return f'{self.name} ({self.value})'

    def __repr__(self) -> str:
        return self.value


# TODO: add class for each slot/position?


class ConnectFour:
    #region Attributes and Printing
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
                    piece_str = "|" + piece_str.strip() + "|"
                row.append(piece_str)
            result.append("|" + "|".join(row) + "|")

        result.append(row_div)
        final_row = []
        for i in range(self.columns):
            final_row.append(f" {i+1} ")
        result.append("|" + "|".join(final_row) + "|")
        return "\n".join(result) + "\n"

    def __str__(self) -> str:
        """Return a basic string representation of the board without any highlighted positions."""
        return self.highlighted_str()

    #endregion

    def insert(self, col: int) -> int:
        if col not in range(self.columns):
            raise OutOfBoundsError(f'Column {one_index(col)} is out of bounds.')

        row = self.first_available_row(col)
        if row == -1:
            raise InvalidInsertError(f"Column {one_index(col)} is full.")

        self.board[row][col] = self.player_whose_turn_it_is()
        self.current_turn += 1
        if not self.pretend:
            self.log.normal(f'{self.board[row][col]} played a piece in column {one_index(col)}.')
        return row

    def available_columns(self) -> list[int]:
        return [c for c in range(self.columns) if self.board[0][c] is Piece.EMPTY]

    def first_available_row(self, col: int) -> int:
        if col not in range(self.columns):
            raise OutOfBoundsError(f'Column {one_index(col)} is out of bounds.')
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] is Piece.EMPTY:
                return r
        return -1

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
                scores[c] += self.window_score(window, target_player = player, target_position = (r,c))
            
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
    
    def window_score(self, positions: list[tuple], target_player: Piece = None, target_position: tuple = None) -> int:
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
            shape = "–" if is_horizontal else "|"
            if is_diagonal:
                shape = "/" if change_r < 0 else "\\"
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
                        if (predicted_winner := prediction.board_winner()) is not Piece.EMPTY:
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
            best_cols = [i for i, s in enumerate(scores) if s is best_score and i in available]
            col = random.choice(best_cols)
        self.log.verbose(f'get_best_move({player}):', f'column {one_index(col)}')
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
                self.log.debug("board_winner:", self.winner, window)
                return self.winner
        self.winner = Piece.EMPTY
        self.winning_group = []
        return self.winner

    def board_has_winner(self):
        return self.board_winner() != Piece.EMPTY

    def game_over(self):
        return self.board_full() or self.board_has_winner()

    #endregion

    #region Turn Mechanics
    def play(self, turns = None) -> bool:
        max_turns = self.rows * self.columns if turns is None else min(self.rows * self.columns, self.current_turn + turns)
        human_players: list[Piece] = [Piece.RED, Piece.BLACK] if self.players == 2 else [self.human_player]
        row, col, attempt = -1, -1, 1
        while not self.game_over() and self.current_turn <= max_turns:
            # Print the board if this is a real game.
            if not self.pretend and attempt == 1:
                self.log.normal(self.highlighted_str([(row, col)]))

            # Get the player's move choice (which column they'll put a piece into).
            if self.player_whose_turn_it_is() in human_players:
                col = self.prompt_for_column_choice()
            else:
                col = self.get_best_current_move()

            # Attempt the move
            try:
                row = self.insert(col)
                attempt = 1
            except Exception as e:
                self.log.error(e)
                self.log.normal("Please try again.")
                attempt += 1

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


def main() -> None:
    # game = ConnectFour.new(players=1, log_level=LogLevel.DEBUG)
    game = ConnectFour.new()
    game.play()


if __name__ == "__main__":
    main()
