# 🤖 Sai — Rule-Based AI Chatbot

**Project 1** of the DecodeLabs AI Engineering Internship (Batch 2026).

Sai is a simple **rule-based chatbot** built in Python. It doesn't use any
machine learning — instead, it relies on clean control flow (dictionary
lookups + fallback logic) to simulate basic conversation. This project lays
the foundation for understanding how deterministic "guardrail" logic works
before moving on to probabilistic, LLM-based systems.

## ✨ Features

- **Continuous input loop** — chats with you until you say goodbye
- **Input sanitization** — normalizes case and whitespace before processing
- **Dictionary-based knowledge base** — fast O(1) intent lookup instead of a
  long `if-elif` ladder
- **Partial-match matching** — recognizes intents even inside longer sentences
- **Randomized responses** — picks from multiple replies for a more natural feel
- **Fallback handling** — gracefully responds when it doesn't understand
- **Clean exit strategy** — type `bye`, `exit`, or `quit` to leave

## 🛠️ Tech Stack

- Python 3.6+
- No external dependencies (standard library only)

## 🚀 Getting Started

### Prerequisites
Make sure Python 3 is installed:
```bash
python3 --version
```

### Installation
Clone this repository:
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### Run the chatbot
```bash
python3 chatbot.py
```
On Windows, use `python chatbot.py` if `python3` isn't recognized.

## 💬 Example Conversation

```
🤖 Sai: Hello! I'm your Project 1 rule-based chatbot.
🤖 Sai: Type 'bye' / 'exit' / 'quit' anytime to leave.

You: hello
🤖 Sai: Hi there! How can I help you today?

You: what is your name
🤖 Sai: I'm Sai, your friendly rule-based chatbot.

You: thank you
🤖 Sai: Anytime!

You: bye
🤖 Sai: Goodbye! Have a great day. 👋
```

## 📂 Project Structure

```
.
├── chatbot.py     # Main chatbot script
└── README.md      # Project documentation
```

## 🧠 How It Works

1. **Sanitization** — raw input is lowercased and stripped of whitespace.
2. **Intent matching** — the cleaned input is checked against a dictionary
   of known intents (exact match first, then partial match).
3. **Response generation** — a matching intent returns a random reply from
   its list; no match triggers a fallback response.
4. **Loop control** — the process repeats inside a `while True` loop until
   an exit command (`bye`, `exit`, `quit`) breaks the cycle.

## 🔮 Future Improvements

- Expand the vocabulary with more intents
- Add nested conditions for context-aware replies
- Give Sai a more distinct personality
- Hybrid mode: fall back to an LLM when no rule matches

## 📜 License

This project was built for educational purposes as part of the DecodeLabs
Industrial Training Kit.

## 📞 Contact

**DecodeLabs**
📧 decodelabs.tech@gmail.com
🌐 [www.decodelabs.tech](https://www.decodelabs.tech)
📍 Greater Lucknow, India
