"""The ConnectFour class allows users to create and configure an instance of single game of connect four that can be played."""
import random
from copy import deepcopy
from typing import Callable, Union

from codetiming import Timer

from . import errors, utils

# Internal module imports
from .logger import Logger, LogLevel
from .piece import Piece

# region Globals
POINTS_PER_PIECE = 2
WINNING_SCORE = 1000
OPPONENT_SCORE_MODIFIER = 1.25
OBVIOUS_MOVE_THRESHOLD = 500
ALPHA_BETA_START = 10000
# endregion


class ConnectFour:
    """
    ConnectFour is a class dedicated to constructing and playing a game of connect four.

    It also implements a minimax algorithm with alpha-beta pruning to play against an AI.
    """

    # region Attributes and Printing
    rows: int = 6
    columns: int = 7
    slots: list[tuple[int]] = []
    slots_by_position: dict[tuple[int], list] = {}

    def __init__(
        self,
        players: int = 0,
        human_player: Piece = None,
        board: list[list[Piece]] = None,
        pretend: bool = False,
        log_level: Union[LogLevel, str] = LogLevel.NONE,
        rows: int = 6,
        columns: int = 7,
        minimax_results: dict[str, int] = {},
    ) -> None:
        """
        Initialize a ConnectFour game based on the configuration parameters provided.

        Args:
            players (int): The number of players in the game.
            human_player (Piece, optional): The piece assigned to the human player. Defaults to None.
            board (list[list[Piece]], optional): The initial game board. Defaults to None.
            pretend (bool, optional): Whether to simulate the game without making actual moves. Defaults to False.
            log_level (Union[LogLevel, str], optional): The log level for game logging. Defaults to LogLevel.NONE.
            rows (int, optional): The number of rows in the game board. Defaults to 6.
            columns (int, optional): The number of columns in the game board. Defaults to 7.
            minimax_results (dict[str, int], optional): The results of the minimax algorithm. Defaults to {}.
        """
        self.log: Logger = Logger(LogLevel[log_level]) if isinstance(log_level, str) else Logger(log_level)
        self.pretend: bool = pretend
        self.players: int = players
        self.human_player: Piece = (
            self.prompt_for_color_choice() if human_player is None and self.players == 1 else human_player
        )
        self.rows: int = rows
        self.columns: int = columns
        self.board: list[list[Piece]] = board or [[Piece.EMPTY for c in range(self.columns)] for r in range(self.rows)]
        self.starting_player: Piece = Piece.RED
        self.current_turn: int = self.determine_turn()
        self.winning_group: list[tuple] = []
        self.winner: Piece = Piece.EMPTY
        self.slots: list[tuple[int]] = self.get_slots()
        self.slots_by_position: dict[tuple[int], list[tuple[int]]] = self.get_slots_by_position()
        self.minimax_results: dict[str, int] = minimax_results
        self.plays: list[tuple[int]] = []

    @classmethod
    def new(cls, **kwargs) -> "ConnectFour":
        """
        Instantiate a new instance of the ConnectFour class prompting for the number of players by default.

        Use the built-in constructor if you are providing the number of players via code.
        """
        instance = cls(**kwargs)
        if "players" not in kwargs:
            instance.players = instance.prompt_for_number_players()
        return instance

    @staticmethod
    def lowest_row(positions) -> int:
        """Return the lowest row in the given set of board positions."""
        result = -1
        for r, c in positions:
            if r > result:
                result = r
        return result

    @staticmethod
    def opponent(player: Piece) -> Piece:
        """
        Return the opponent of the given piece.

        There are only two piece choices, so this will be obvious.
        If None is provided, None will be returned.
        """
        if player is Piece.BLACK:
            return Piece.RED
        elif player is Piece.RED:
            return Piece.BLACK
        return None

    @staticmethod
    def is_winning_score(score: int) -> bool:
        """Return True if the provided score is considered a win, otherwise False."""
        return score == -WINNING_SCORE or score == WINNING_SCORE

    def determine_turn(self) -> int:
        """Return the current turn number based on how many positions on the board are already occupied."""
        occupied_positions = 0
        for c in range(self.columns):
            for r in range(self.rows - 1, -1, -1):
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
        """
        Return a string representation of the shape of the set of adjacent positions on the board.

        This is intended to quickly express whether a set of positions is:
        vertical (|), horizontal (-), diagonal ascending (/), or diagonal descending (\).
        """
        change_r = positions[1][0] - positions[0][0]
        change_c = positions[1][1] - positions[0][1]
        shape = "-" if change_r == 0 else "|"
        if change_r != 0 != change_c:
            shape = "/" if change_r < 0 else "\\"
        return shape

    @classmethod
    def get_slots(cls):
        """Return all possible winning board positions (four in a row, adjacent)."""
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
                # - group that includes r.
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
        """
        Return a dictionary of all possible winning board positions by member slot.

        Keys are tuples indicating row and column.
        Values are list containing the three other positions that would make up that four-in-a-row set of positions.
        """
        result = {}
        cls.slots = cls.slots or cls.get_slots()
        for c in range(cls.columns):
            for r in range(cls.rows):
                p = (r, c)
                result[p] = []
                for s in cls.slots:
                    if p in s:
                        result[p].append([new_p for new_p in s if new_p != p])
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
                    result.append(f"{r}{c}{self.board[r][c].value}")
        return ",".join(result)

    # endregion

    # region Column and Row Interactions
    def insert(self, col: int, player: Piece = None) -> int:
        """
        Return the row into which a piece has been inserted.

        If the column provided is invalid or full, an error will be raised.
        """
        if col not in range(self.columns):
            raise errors.OutOfBoundsError(f"insert(): column {utils.utils.one_index(col)} is out of bounds.")
        row = self.first_available_row(col)
        if row == -1:
            raise errors.InvalidInsertError(f"insert(): column {utils.one_index(col)} is full.")
        if player is None:
            player = self.player_whose_turn_it_is()

        self.board[row][col] = player
        self.current_turn += 1
        self.plays.append((row, col))
        if not self.pretend:
            self.log.normal(f"{self.board[row][col]} played a piece in column {utils.one_index(col)}.")
        return row

    def remove(self, col: int) -> None:
        """
        Remove the topmost piece from the board at the provided column.

        An error will be raised if there are no pieces in that column or if the column is not valid.
        Since removing a piece could invalidate a previously valid win,
        `evaluate_board_win` will be called after removal to reset the winner status.
        """
        if col not in range(self.columns):
            raise errors.OutOfBoundsError(f"remove(): column {utils.one_index(col)} is out of bounds.")

        for row in range(self.rows):
            if self.board[row][col] is not Piece.EMPTY:
                old_piece = self.board[row][col]
                self.board[row][col] = Piece.EMPTY
                self.current_turn -= 1
                if (row, col) not in self.plays:
                    self.plays.remove((row, col))
                self.evaluate_board_win()
                if not self.pretend:
                    self.log.error(
                        f"{old_piece} removed their piece from column {utils.one_index(col)};",
                        f"Current turn is now {self.current_turn}",
                    )
                return
        raise errors.InvalidRemoveError(f"remove(): there are no pieces in column {utils.one_index(col)} to remove.")

    @classmethod
    def prioritize_central_columns(cls, col) -> int:
        """Return weights indicating preferability of columns. Higher weights for indexes closer to the midpoint."""
        mid = cls.columns / 2
        return round(abs((mid - 0.5) - col))

    def available_columns(self) -> list[int]:
        """
        Return a list of column indexes whose columns are not yet full. This represents valid moves.

        The result will be sorted such that indexes closer to the midpoint are listed first.
        """
        available = [c for c in range(self.columns) if self.board[0][c] is Piece.EMPTY]
        available.sort(key=self.prioritize_central_columns)
        return available

    def first_available_row(self, col: int) -> int:
        """Return the first row index that is available in the current column."""
        if col not in range(self.columns):
            raise errors.OutOfBoundsError(f"Column {utils.one_index(col)} is out of bounds.")
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] is Piece.EMPTY:
                return r
        return -1

    def available_spots_in_columns(self) -> list[int]:
        """
        Return a list of integers representing the number of available spots in each column.

        The result is the same as the row number of the last occupied position).
        """
        last_occupied_positions = [self.rows] * self.columns
        for c in range(self.columns):
            for r in range(self.rows):
                if self.board[r][c] is not Piece.EMPTY:
                    last_occupied_positions[c] = r
                    break
        return last_occupied_positions

    # endregion

    # region Minimax Algorithm
    def slot_score(self, positions: list[tuple[int, int]], win_only: bool = False) -> float:
        """
        Return the score of the provided list of four board positions.

        The score will be based on how many pieces a single player has in a row.
        """
        MINIMUM_PIECES_TO_SCORE = 1
        # Count occurrences
        total = {Piece.RED: 0, Piece.BLACK: 0, Piece.EMPTY: 0}
        for r, c in positions:
            total[self.board[r][c]] += 1
            # If both players are represented or if there's an empty and we're only looking for a full group, return 0.
            if (win_only and self.board[r][c] is Piece.EMPTY) or (total[Piece.BLACK] > 0 and total[Piece.RED] > 0):
                return 0
        # At this point we know that, at most, one player has pieces in this group.
        # Determine which piece to count for the score.
        piece, sign = (Piece.RED, 1) if total[Piece.BLACK] == 0 else (Piece.BLACK, -1)
        # Winning score. (If win_only is True, then this is automatically a winner).
        if win_only or total[piece] == 4:
            return sign * (WINNING_SCORE)

        # If the minimum number of player pieces hasn't been reached, return 0.
        # if total[Piece.EMPTY] > len(positions)-MINIMUM_PIECES_TO_SCORE:
        #     return 0
        # Non-winning score
        return sign * (POINTS_PER_PIECE ** (total[piece]))

    def move_scores(
        self,
        player: Piece = None,
        available: list[tuple[int]] = None,
        win_only: bool = False,
    ) -> list[int]:
        """
        Return a board score for every column reflecting how advantageous it is for either player.

        The score will be positive if it favors Piece.RED, negative if it favors Piece.BLACK, or 0 if there's no obvious
        benefit to either.
        """
        t = Timer(name="\tmove_score()", text="{name} took {:.3f}s", logger=self.log.debug)
        t.start()
        if available is None:
            available = self.available_columns()
        if player is None:
            player = self.player_whose_turn_it_is()
        opponent = self.opponent(player)
        game = self
        if not self.pretend:
            game = ConnectFour(
                board=deepcopy(self.board),
                players=0,
                pretend=True,
                log_level=self.log.level,
                rows=self.rows,
                columns=self.columns,
            )
        scores = [0] * self.columns
        opponent_scores = [0] * self.columns
        for c in available:
            if win_only:
                scores[c] = self.winning_move_score(c)
            else:
                game.insert(c, player)
                scores[c] = game.evaluate_board_outlook(player)
                game.remove(c)
                game.insert(c, opponent)
                opponent_scores[c] = game.evaluate_board_outlook(opponent)
                game.remove(c)
        for c in available:
            scores[c] -= int(opponent_scores[c] / OPPONENT_SCORE_MODIFIER)
        t.stop()
        return scores

    def winning_move_score(self, col: int) -> bool:
        """
        Return an integer representing which player would win if a move was made in the given column.

        One ("1") if RED would win
        Negative One ("-1") if BLACK would win
        Zero ("0") means neither player would win.
        """
        try:
            row = self.first_available_row(col)
            for s in self.slots_by_position[(row, col)]:
                for i in range(1, len(s)):
                    r, c = s[i - 1]
                    prev = self.board[r][c]
                    if prev is Piece.EMPTY:
                        break
                    r, c = s[i]
                    curr = self.board[r][c]
                    if curr is Piece.EMPTY or prev is not curr:
                        break
                    if i == len(s) - 1:
                        return 1 if curr is Piece.RED else -1
        except Exception:
            pass
        return 0

    def evaluate_board_win(self) -> int:
        """Determine if there is a winner, return the corresponding score, otherwise 0."""
        for s in self.slots:
            score = self.slot_score(s, win_only=True)
            if self.is_winning_score(score):
                r, c = s[0]
                self.winning_group = s
                self.winner = self.board[r][c]
                return score
        self.winning_group = []
        self.winner = Piece.EMPTY
        return 0

    def evaluate_board_outlook(self, player: Piece = None) -> int:
        """Return the overall outlook (likelihood of victory) of the current board for the current/provided player."""
        result = 0
        for group in self.slots:
            score = self.slot_score(group)
            result += score
            if self.is_winning_score(score):
                return int(score)
        return int(result)

    def minimax(
        self,
        player: Piece = None,
        depth: int = 5,
        alpha: int = -ALPHA_BETA_START,
        beta: int = ALPHA_BETA_START,
    ) -> int:
        """
        Return the score of the current board.

        Positive return value indicate an advantage for the 'max' player, RED; negative favors the 'min' player, BLACK.
        The score will be based on a simple score of how hypothetically beneficial each move is to each player.
        This function will be called recursively to a specified depth to incorporate how future 'branches'
        of moves should influence the score for the current board.

        NOTE: This function is computationally intensive. A depth > 5 may result in unreasonably long wait times.
        """
        VERBOSE_LOG_MAX_INDENTS = 6
        if player is None:
            player = self.player_whose_turn_it_is()
        if not self.pretend:
            self.log.error("minimax() is running on a real board. You may see log messages for hypothetical moves.")
        is_max = player is Piece.RED
        score = 0
        string_repr = repr(self)

        # If we've seen this board before and we're at a leaf, use the previously calculated score.
        if string_repr in self.minimax_results and (
            self.is_winning_score(self.minimax_results[string_repr]) or depth < 1
        ):
            score = self.minimax_results[string_repr]
        else:
            # Otherwise, check for a winner.
            score = self.evaluate_board_win()
        # If the board outcome has been determined or the intended depth has been reached, evaluate score and return.
        # Otherwise, make recursive call to go deeper.
        if self.is_winning_score(score):
            self.minimax_results[string_repr] = score
            # Increase the absolute value of this win score by the current.
            # This ensures that earlier wins are weighted more than later wins.
            score += depth if score > 0 else -depth
            self.log.verbose(
                "\t" * (VERBOSE_LOG_MAX_INDENTS - depth),
                f"{repr(self.winner)} ({depth}) win:",
                score,
            )
        elif self.board_is_full():
            return 0
        elif depth < 1:
            # Calling self.evaluate_board_outlook() is appropriate here, but computationally intensive.
            # One alternative is to set the score to 0 since it's not a win.
            score = self.evaluate_board_outlook()
            self.minimax_results[string_repr] = score
        else:
            sign = 1 if is_max else -1
            best = sign * -(ALPHA_BETA_START * 10)
            # Reduce the available options to only the obvious ones (win or block), if available.
            # A reasonable player wouldn't consider any others.
            available = self.available_columns()
            obvious = []
            for col in available:
                if self.winning_move_score(col) != 0:
                    obvious.append(col)
            if len(obvious) > 0:
                available = obvious
            # Evaluate the game outlook based on each possible move, selecting the most desirable one.
            for col in available:
                self.log.verbose(
                    "\t" * (VERBOSE_LOG_MAX_INDENTS - depth),
                    f"{repr(player)} ({depth}) playing in col {col} on turn {self.current_turn},",
                    f"alpha: {alpha}, beta: {beta}",
                )
                self.insert(col)
                val = self.minimax(
                    player=self.opponent(player),
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                )
                self.remove(col)

                if (is_max and val > best) or (not is_max and val < best):
                    best = val

                    if is_max and best > alpha:
                        alpha = best
                    elif not is_max and best < beta:
                        beta = best

                if beta <= alpha:
                    self.log.verbose(
                        "\t" * (VERBOSE_LOG_MAX_INDENTS - depth),
                        f"PRUNING because {repr(player)} ({depth}) playing in col {col} on turn {self.current_turn}",
                        f"would result in {val} (best: {best}), alpha: {alpha} beta: {beta}",
                    )
                    break
                self.log.verbose(
                    "\t" * (VERBOSE_LOG_MAX_INDENTS - depth),
                    f"{repr(player)} ({depth}) playing in col {col} on turn {self.current_turn}",
                    f"would result in {val} (best: {best}), alpha: {alpha} beta: {beta}",
                )
            score = best
            # self.log.normal(self, f"\n{repr(player)} ({depth}), {score}")
        return score

    def get_best_move(self) -> int:
        """
        Return the best column number for the current player to insert their piece.

        This evaluation will involve looking for obvious wins or blocks, running the minimax algorithm to pursue optimal
        future outcomes, and weighting central moves over outer edge moves.
        If an unambiguous move is identified, it will be immediately returned, avoiding unnecessary computation.
        """
        if self.board_is_full():
            raise errors.BoardFullError("get_best_move: cannot evaluate the best move of a full board.")
        player = self.player_whose_turn_it_is()
        is_max = player is Piece.RED
        minimax_label = "MAX" if is_max else "MIN"
        min_or_max_func = max if is_max else min
        available = self.available_columns()
        minimax_depth = 7
        self.log.debug("get_best_move:", repr(player), minimax_label)

        # If there's only one available move, use it.
        if len(available) == 1:
            self.log.debug("\tonly move:", available[0])
            return available[0]

        # Determine if there's an obvious choice to win or block.
        immediate_win_block_scores = [0] * self.columns
        only_obvious_move_scores = []
        for col in available:
            immediate_win_block_scores[col] = self.winning_move_score(col)
            if immediate_win_block_scores[col] != 0:
                only_obvious_move_scores.append(immediate_win_block_scores[col])
        self.log.debug("\twin/block scores:", immediate_win_block_scores)
        if len(only_obvious_move_scores) > 0:
            obvious_move_score = min_or_max_func(only_obvious_move_scores)
            for col in available:
                if immediate_win_block_scores[col] == obvious_move_score:
                    self.log.debug("\twin/block move:", col)
                    return col

        # Get the minimax score for every possible move.
        adjusted_depth = min(self.current_turn - 1, minimax_depth)
        minimax_scores = [0] * self.columns
        if adjusted_depth > 0:
            t = Timer(
                name=f"\tminimax(depth: {adjusted_depth})",
                text="{name} took {:.3f}s",
                logger=self.log.debug,
            )
            t.start()
            future = self
            if not self.pretend:
                future = ConnectFour(
                    players=0,
                    board=deepcopy(self.board),
                    pretend=True,
                    log_level=self.log.level,
                    rows=self.rows,
                    columns=self.columns,
                    minimax_results=self.minimax_results,
                )
            alpha, beta = -ALPHA_BETA_START, ALPHA_BETA_START
            for col in available:
                future.insert(col)
                self.log.verbose(
                    "\t_",
                    f"{repr(player)} ({adjusted_depth+1}) playing in col {col} on turn {self.current_turn},",
                    f"alpha: {alpha}, beta: {beta}",
                )
                val = future.minimax(
                    player=future.opponent(player),
                    depth=adjusted_depth,
                    alpha=alpha,
                    beta=beta,
                )
                minimax_scores[col] = val
                future.remove(col)
                self.log.verbose(
                    "\t_",
                    f"{repr(player)} ({adjusted_depth+1}) playing in col {col} on turn {self.current_turn}",
                    f"would result in {val}, alpha: {alpha}, beta: {beta}",
                )
            t.stop()
        best_minimax_columns = self.best_columns_from_scores(minimax_scores, min_or_max_func, available=available)
        self.log.debug("\tminmax scores:", minimax_scores, best_minimax_columns)
        if len(best_minimax_columns) == 1:
            self.log.debug(
                "\tbest minimax move:",
                f"{best_minimax_columns[0]} ({minimax_scores[best_minimax_columns[0]]})",
            )
            return best_minimax_columns[0]

        # Evaluate the board for the best immediate moves.
        simple_scores = self.move_scores(player=player, available=available)
        self.log.debug("\tmove scores:", simple_scores)

        # If multiple moves have the same minimax score,
        # see how many of those moves are shared by the best simple immediate moves.
        weighted_scores = self.weighted_scores(scores=simple_scores, weights=minimax_scores)
        best_columns = self.best_columns_from_scores(weighted_scores, min_or_max_func, available=available)
        self.log.debug(
            "\tweighted scores:",
            f"{weighted_scores} {best_columns} {[(simple_scores[c], minimax_scores[c]) for c in best_columns]}",
        )
        if len(best_columns) == 1 or len(best_columns) >= 4:
            self.log.debug(
                "\tweighted move:",
                f"{best_columns[0]} ({minimax_scores[best_columns[0]]}, {simple_scores[best_columns[0]]})",
            )
            return best_columns[0]

        random_choice = random.choice(best_columns)
        self.log.debug("\trandom best move:", random_choice)
        return random_choice

    def best_columns_from_scores(
        self,
        scores: list[int],
        best_func: Callable[[list[int]], int] = max,
        available: list[int] = None,
    ) -> int:
        """
        Return all the columns that share the best score in the provided list as evaluated by the 'best_func' function.

        Optionally limit the returned items by the provide set of available columns.

        NOTE: This will not return a sorted list of columns ordered in accordance with their score.
        """
        if available is None:
            available = self.available_columns()
        best_score = best_func([s for c, s in enumerate(scores) if c in available])
        best_available = [c for c, s in enumerate(scores) if s == best_score and c in available]
        best_available.sort(key=self.prioritize_central_columns)
        return best_available

    def weighted_scores(self, scores: list[int], weights: list[int]) -> int:
        """Return an updated array of scores for each column, averaging the provided score and corresponding weight."""
        weighted: list[int] = []
        for c, s in enumerate(scores):
            weight = weights[c] if c < len(weights) else 0
            weighted.append(int((s + weight) / 2))
        return weighted

    # endregion

    # region Game Completion
    def board_is_full(self) -> bool:
        """Return True if the board has no more available columns, otherwise False."""
        return len(self.available_columns()) == 0

    def board_has_winner(self):
        """Return True if there is a winning score on the board."""
        return self.is_winning_score(self.evaluate_board_win())

    def game_over(self):
        """Return True if the board is either full or has a winner, otherwise False."""
        return self.board_has_winner() or self.board_is_full()

    # endregion

    # region Turn Mechanics
    def play(self, turns: int = None) -> bool:
        """
        Play the game for the provided number of turns (or as many as remain if not provided).

        The game will prompt for moves if it is configured for human players,
        otherwise it will use self.get_best_move to play for non-human players.

        Returns True if the plays occurred successfully, or False if there was a problem during play.
        """
        if turns is None:
            turns = self.rows * self.columns
        human_players: list[Piece] = (
            [Piece.RED, Piece.BLACK] if self.players == 2 else [self.human_player] * self.players
        )

        # Print the board if this is a real game.
        if not self.pretend:
            self.log.normal(self)
        row, col, attempt = -1, -1, 1
        turn_counter = 0
        while not self.game_over() and turn_counter < turns:
            # Get the player's move choice (which column they'll put a piece into).
            if self.player_whose_turn_it_is() in human_players:
                col = self.prompt_for_column_choice()
            else:
                col = self.get_best_move()

            # Attempt the move
            try:
                row = self.insert(col)
                turn_counter += 1
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
        self.log.debug(f"play: finished playing {turn_counter} turns.")
        # If this is a real game and the game has ended, print the results.
        if not self.pretend and self.game_over():
            highlight = self.winning_group if len(self.winning_group) > 0 else []
            self.log.normal(self.highlighted_str(highlight))
            self.log.normal("Game finished!")
            if self.winner in [Piece.RED, Piece.BLACK]:
                self.log.normal("Winner:", self.winner)
            else:
                self.log.normal("Stalemate")
        return True

    def prompt_for_color_choice(self) -> Piece:
        """Prompt the user to provide a color/piece choice via stdin."""
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
        """Prompt the user for a column choice. Columns are adjusted to be one-indexed for ease of use."""
        col, request = -1, 0
        while col not in range(self.columns):
            col = input(
                f"{self.player_whose_turn_it_is()}, please select a column number (1-{utils.one_index(self.columns)}):",
                "\n",
            )
            try:
                col = int(col) - 1
                request += 1
            except errors.InvalidInsertError:
                self.log.normal("Invalid column.")
                if request > 5:
                    raise Exception("Too many failed attempts")
        return col

    def prompt_for_number_players(self) -> int:
        """Prompt for a number of human players for this game."""
        players = -1
        request = 0
        max_attempts = 5
        while players not in [0, 1, 2] and request < max_attempts:
            if request > 0:
                self.log.normal(f"Invalid selection. {request} of 5 attempts.")
            players = input("How many human players would like to participate?\n")
            try:
                players = int(players)
                if players not in [0, 1, 2]:
                    raise errors.InvalidPlayersError(f"{players} is not a valid option.")
            except Exception as e:
                self.log.error(e)
            request += 1
        self.log.info(f"{players} human players.")
        return players

    # endregion
