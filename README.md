# Playing Hangman Game with High Success Rate Using Bidirectional LSTM

We study strategies playing Hangman game with high successful rate (~ 60%) using bidirectional LSTM.

## What is Hangman?

Hangman is a classic word-guessing game where players try to discover a hidden word by guessing individual letters. Here's how it works:

1. **Setup**: A secret word is chosen and displayed as a series of blank spaces (underscores), one for each letter
2. **Guessing**: Players guess one letter at a time
3. **Correct Guess**: If the letter appears in the word, all instances are revealed in their correct positions
4. **Wrong Guess**: If the letter doesn't appear, it counts as a mistake (traditionally drawn as part of a hanging stick figure)
5. **Win Condition**: Player wins by revealing the entire word before making too many wrong guesses
6. **Lose Condition**: Player loses after 6 wrong guesses

### Example Game Flow:
```
Secret word: "PYTHON"
Display: _ _ _ _ _ _

Guess 'E': Not in word (1 wrong)
Display: _ _ _ _ _ _

Guess 'T': In word!
Display: _ _ T _ _ _

Guess 'H': In word!
Display: _ _ T H _ _

Guess 'P': In word!
Display: P _ T H _ _

Guess 'Y': In word!
Display: P Y T H _ _

Guess 'O': In word!
Display: P Y T H O _

Guess 'N': In word! YOU WIN!
Display: P Y T H O N
```

## Overview

This project demonstrates how to build deep learning approaches for playing Hangman game with high successful rate, and discusses potential future improvements.

## Architecture

### Multi-Stage Strategy

- **Early Stage** (0-1 letters known): Statistical frequency analysis
- **Mid Stage** (2+ letters, many candidates): Hybrid frequency + bidirectional LSTM  
- **Late Stage** (few candidates or critical decisions): bidirectional LSTM

## Performance

- **Overall Success Rate**: 48-60% (varies by model size)
- **Training Data**: 300K English words from public sources
- **Testing Dara**: 1k English words disjoint from training data

## Files

- `hangman_public.ipynb` - Main notebook with complete implementation
- `words_public.txt` - Public English word dataset (370K+ words) - Downloaded from https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt

## Contributing

Feel free to fork, experiment, and submit improvements! Areas for contribution:
- New model architectures
- Alternative training strategies  
- Performance optimizations
- Additional analysis and visualizations
