# Playing Hangman Game with High Success Rate Using Bidirectional LSTM

We study strategies playing Hangman game with high success rate (~ 60%) using bidirectional LSTM.

## What is Hangman?

Hangman is a classic word-guessing game where players try to discover a hidden word by guessing individual letters:

1. Setup: A secret word is chosen and displayed as blank spaces (underscores)
2. Guessing: Players guess one letter at a time
3. Correct Guess: If the letter appears in the word, all instances are revealed
4. Wrong Guess: If the letter doesn't appear, it counts as a mistake
5. Win Condition: Player wins by revealing the entire word before making 6 wrong guesses
6. Lose Condition: Player loses after 6 wrong guesses

### Example Game (Win):
```
Secret word: "MACHINE"
Initial:     _ _ _ _ _ _ _

Guess 'E': (true)  → _ _ _ _ _ _ E
Guess 'A': (true)  → _ A _ _ _ _ E  
Guess 'T': (false)  → _ A _ _ _ _ E  (1/6 wrong)
Guess 'I': (true)  → _ A _ _ I _ E
Guess 'N': (true)  → _ A _ _ I N E
Guess 'M': (true)  → M A _ _ I N E
Guess 'C': (true)  → M A C _ I N E
Guess 'H': (true)  → M A C H I N E → Win
```


### Example Game (Lose):
```
Secret word: "APPLE"
Initial:     _ _ _ _ _

Guess 'E': (true)  → _ _ _ _ E
Guess 'I': (false)  → _ _ _ _ E  (1/6 wrong)
Guess 'T': (false)  → _ _ _ _ E  (2/6 wrong)
Guess 'H': (false)  → _ _ _ _ E  (3/6 wrong)
Guess 'P': (true)  → _ P P _ E
Guess 'J': (false)  → _ P P _ E  (4/6 wrong)
Guess 'C': (false)  → _ P P _ E  (5/6 wrong)
Guess 'S': (false)  → _ P P _ E  (6/6 wrong) -> Lose
```

## Overview

This project demonstrates how to build deep learning approaches for playing Hangman game with high success rate, and discusses potential future improvements. See the notebook for all the details.

### Strategy: Multi-Stage Solution

- **Early Stage** (0-1 letters known): Statistical frequency analysis
- **Mid Stage** (2+ letters, many candidates): Hybrid frequency + bidirectional LSTM  
- **Late Stage** (few candidates or critical decisions): bidirectional LSTM

## Performance

- **Overall Success Rate**: 48-60% (varies by model size)
- **Training Data**: ~300K English words from public sources
- **Testing Data**: ~ 70k English words disjoint from training data
