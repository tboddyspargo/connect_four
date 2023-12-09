# Overview

This Connect Four game has a very simple AI and CLI which allows humans to play against each other or against the AI. I wrote this game as part of a project on [Codecademy](https://www.codecademy.com/courses/connect-four/). I went significantly beyond the scope that was laid out for my own edification, giving it an AI and the ability to play against itself as well as trying to practice object oriented programming (OOP) principles.

# Requirements

| Python Version | 3.9+ |
| -------------- | ---- |

**NOTE:** _Although this package may work with older versions of python, it has not been tested with them._

# Installation

I have not published this package to `pip`. So, you'll need to download the repo locally in order to install it.
Follow these steps to install this package locally:

```bash
cd ./repo_location/connect_four/
python -m build
```

# Usage

## CLI

Once installed, you can run the game using the provided `c4` script which will be placed into your python environment's `bin` directory. This script will initialize and start a game of connect_four with default options.

```bash
c4
```

## Module

Once installed, you can import the module using the following command:

```python
from connect_four import *
```

This will import all the necessary classes for you to interact with. You may not need them all, so consider what you will use.

## Classes

### ConnectFour

The `ConnectFour` class contains most of the logic for playing a game of connect four. You can initialize a new game using the convenience method `new()` or call the constructor directly. To start playing the game, call the `play()` method.

_Default_

```python
game = ConnectFour.new()
game.play()
```

_Custom_

```python
game = ConnectFour(players = 1, human_player = Piece.BLACK)
game.play()
```

### LogLevel

The `LogLevel` class is a simple comparable enum to allow different levels of detail to supersede each other.

Levels include: `NONE`, `INFO`, `DEBUG`, `VERBOSE`

### Logger

The `Logger`class is facilitates consistently formatted log messages that will respect the provided `LogLevel`. The methods on this class are used to log different kinds of messages (`normal`, `error`, `info`, `debug`, `verbose`).

### Piece

The `Piece` class is an enum that represents a connect four token or the lack of one. The tokens are one of the following:
| Token | Value |
|-------| ------|
|RED|`"X"`|
|BLACK|`"O"`|
|EMPTY|`" "`|

# AI Player

The AI for a computer player is fairly rudimentary, though I have some ideas for improvement. It favors defensiveness, for the most part, and can't see too many moves into the future. It's definitely beatable.

# Development

I used `virtualenvironment` to develop this module, so my instructions may make certain assumptions based on that.

# TODO

- [x] Implement minimax algorithm with alpha-beta pruning to improve AI.
- [ ] Implement `Difficulty` configuration to make the AI dumber/smarter.
- [ ] (Maybe) Implement a `Position` class that can be empty so that `Piece` can be only `RED` or `BLACK`. Would this improve readability?
- [ ] (Maybe) Remove `Piece` class and simplify to just `"X", "O", " "` strings. Would this reduce complexity?
- [ ] Consider simply constructing a board with a list of empty lists (rather than filling it with empties). An empty column has `len == 0`, while a full one has `len == self.rows`. This may improve both time and space efficiency.
