# Gomoku AI

A Python implementation of the Gomoku game (Five in a Row) with an AI opponent using the Minimax algorithm with alpha-beta pruning.

## Features

- **Two Game Modes**:
  - Player vs Player
  - Player vs AI

- **Smart AI Opponent**:
  - Minimax algorithm with alpha-beta pruning
  - Pattern recognition for offensive and defensive play
  - Strategic position evaluation
  - Transposition table for optimization

- **Complete Rule Implementation**:
  - Standard 15×15 board
  - First player restrictions for fairness
  - Victory detection in all directions

## Technical Details

- **AI Strategy**:
  - Pattern-based board evaluation
  - Center control prioritization
  - Depth-limited search with time constraints
  - Tactical move prioritization

- **Performance Optimizations**:
  - Focused move generation around existing pieces
  - Move ordering for better pruning
  - Memory caching of previously evaluated positions

