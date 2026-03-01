# Overview

This Connect Four game has a very simple AI and CLI which allows humans to play against each other or against the AI. I wrote this game as part of a project on [Codecademy](https://www.codecademy.com/courses/connect-four/). I went significantly beyond the scope that was laid out for my own edification, giving it an AI and the ability to play against itself as well as trying to practice object oriented programming (OOP) principles.

## Requirements

| Python Version | 3.9+ |
| -------------- | ---- |

**NOTE:** _Although this package may work with older versions of python, it has not been tested with them._

## Installation

I have not published this package to `pip`. So, you'll need to download the repo locally in order to install it.
Follow these steps to install this package locally:

```bash
uv pip install .
```

## Usage

Once installed, you can run the game using the provided `c4` script which will be placed into your python environment's `bin` directory. This script will initialize and start a game of connect_four with default options.

```bash
c4
```

### Local Dev

During development, you can run `c4` using `uv`:

```bash
# The main entrypoint
uv run c4

# The experiment running entrypoint (for a pre-configured AI v AI game).
uv run t4
```

## Contributing

Update the git hooks directory for this repo:

```sh
make git-hooks
```

## TODO

- [x] Implement minimax algorithm with alpha-beta pruning to improve AI.
- [ ] Implement `Difficulty` configuration to make the AI dumber/smarter.
- [ ] (Maybe) Implement a `Position` class that can be empty so that `Piece` can be only `RED` or `BLACK`. Would this improve readability?
- [ ] (Maybe) Remove `Piece` class and simplify to just `"X", "O", " "` strings. Would this reduce complexity?
- [ ] Consider simply constructing a board with a list of empty lists (rather than filling it with empties). An empty column has `len == 0`, while a full one has `len == self.rows`. This may improve both time and space efficiency.
