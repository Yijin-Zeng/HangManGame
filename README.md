# Hangman AI: Deep Learning Strategy

An intelligent Hangman game solver using deep learning techniques to achieve high success rates through multi-stage strategic gameplay.

## 🎮 What is Hangman?

Hangman is a classic word-guessing game where players try to discover a hidden word by guessing individual letters. Here's how it works:

1. **Setup**: A secret word is chosen and displayed as a series of blank spaces (underscores), one for each letter
2. **Guessing**: Players guess one letter at a time
3. **Correct Guess**: If the letter appears in the word, all instances are revealed in their correct positions
4. **Wrong Guess**: If the letter doesn't appear, it counts as a mistake (traditionally drawn as part of a hanging stick figure)
5. **Win Condition**: Player wins by revealing the entire word before making too many wrong guesses
6. **Lose Condition**: Player loses after a fixed number of wrong guesses (typically 6)

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

The challenge lies in choosing letters strategically to maximize information gain while minimizing wrong guesses.

### Why is Hangman Interesting for AI?

Hangman presents several fascinating challenges that make it an excellent problem for artificial intelligence:

1. **Incomplete Information**: You must make decisions with partial knowledge of the target word
2. **Risk Management**: Balance between safe, common letters vs. strategic guesses
3. **Pattern Recognition**: Identify word patterns from letter positions (e.g., "_ING" suggests common word endings)
4. **Adaptive Strategy**: Optimal strategies change as more letters are revealed
5. **Limited Mistakes**: Unlike many games, you have only 6 wrong guesses before losing
6. **Vocabulary Knowledge**: Success requires understanding of language patterns and word frequency

These challenges combine aspects of **game theory**, **natural language processing**, and **sequential decision making** - making Hangman an ideal testbed for AI techniques.

## 🎯 Overview

This project demonstrates how to build an AI that can play Hangman optimally using a combination of statistical analysis and LSTM neural networks. The AI adapts its strategy based on the game state, achieving success rates of 50-80% depending on word length.

## 🏗️ Architecture

### Multi-Stage Strategy
- **Early Stage** (0-1 letters known): Statistical frequency analysis
- **Mid Stage** (2+ letters, many candidates): Hybrid frequency + neural network  
- **Late Stage** (few candidates or critical decisions): Pure pattern recognition

### Configurable LSTM Models
- Small: 128/64 LSTM layers (32 dense)
- Medium: 256/128 LSTM layers (64 dense)  
- Large: 512/256/128 LSTM layers (64 dense)
- XLarge: 512/256/128/64 LSTM layers (128 dense)

## 📊 Performance

- **Overall Success Rate**: 50-70% (varies by model size)
- **Best Performance**: 80%+ success on words with 11+ letters
- **Training Data**: 370K+ English words from public sources
- **Test Games**: 1000+ local simulations per model

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pandas
```

### Usage
1. **Run the notebook**: Open `hangman_public.ipynb` 
2. **Download word data**: The notebook automatically uses public word datasets
3. **Train models**: Compare different model architectures
4. **Simulate games**: Test performance on 1000+ local games

### Quick Example
```python
# Create and train a model
model = create_hangman_model(lstm_layers=[256, 128], dense_dim=64)

# Initialize game simulator  
game = HangmanGame(model, test_words)

# Run 1000 test games
results = game.simulate_games(1000)
success_rate = sum(r['success'] for r in results) / len(results)
print(f"Success rate: {success_rate:.3f}")
```

## 📁 Files

- `hangman_public.ipynb` - Main notebook with complete implementation
- `words_public.txt` - Public English word dataset (370K+ words)
- `best_model/` - Pre-trained model files
- Generated model files: `hangman_model_{size}.keras`

## 🔬 Key Insights

1. **Model Size vs Performance**: Larger models perform better, but with diminishing returns
2. **Word Length Correlation**: Success rate increases significantly with word length
3. **Strategy Adaptation**: Multi-stage approach outperforms single-strategy methods
4. **Pattern Recognition**: LSTM excels when sufficient context is available

## 📈 Results Visualization

The notebook includes comprehensive analysis:
- Success rate comparisons across model sizes
- Performance breakdown by word length
- Training loss curves
- Statistical analysis of guessing patterns

## 🎓 Educational Value

This project demonstrates:
- Text sequence modeling with LSTMs
- Multi-stage decision making in AI
- Model architecture comparison
- Game theory applications in ML
- Statistical vs. learned approaches

## 🔮 Future Improvements

- Specialized models for short words (≤7 letters)
- Adaptive stage transitions based on word length  
- Integration of word frequency information
- Ensemble methods combining multiple strategies

## 📜 License

This project is open source and educational. The word dataset is from public domain sources.

## 🤝 Contributing

Feel free to fork, experiment, and submit improvements! Areas for contribution:
- New model architectures
- Alternative training strategies  
- Performance optimizations
- Additional analysis and visualizations
