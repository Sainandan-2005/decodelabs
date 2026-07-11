"""
Project 1: Rule-Based AI Chatbot
DecodeLabs — Industrial Training Kit (Batch 2026)

Goal: A simple rule-based chatbot that responds to predefined user
inputs using if-else / dictionary logic and runs in a continuous loop.

Specification covered:
  [x] INPUT LOOP     -> continuous while cycle
  [x] SANITIZATION   -> lowercased + stripped input
  [x] KNOWLEDGE BASE -> dictionary with 5+ intents
  [x] FALLBACK       -> default response for unknowns
  [x] EXIT STRATEGY  -> clean break command
"""

import random

BOT_NAME = "Sai"

# ---------------------------------------------------------------------------
# PHASE: KNOWLEDGE BASE
# Each key is an "intent" (a normalized user phrase). Values can be a single
# string or a list of strings (the bot picks one at random for variety).
# ---------------------------------------------------------------------------
responses = {
    "hello": ["Hi there! How can I help you today?", "Hey! Good to see you."],
    "hi": ["Hello!", "Hi! What's on your mind?"],
    "how are you": ["I'm just lines of code, but I'm running smoothly! You?",
                     "Doing great, thanks for asking!"],
    "what is your name": [f"I'm {BOT_NAME}, your friendly rule-based chatbot."],
    "who made you": ["I was built as Project 1 for the DecodeLabs AI Internship."],
    "what can you do": ["I can chat using simple rule-based logic — no ML, just clean control flow!"],
    "thank you": ["You're welcome!", "Anytime!"],
    "thanks": ["No problem!", "Glad to help!"],
    "help": ["Try greeting me, asking my name, or just say 'bye' to leave."],
}

EXIT_COMMANDS = {"bye", "exit", "quit", "goodbye"}
FALLBACK_RESPONSES = [
    "I do not understand that yet. Could you rephrase?",
    "Hmm, I don't have a rule for that one.",
    "I'm still learning — try asking something else!",
]


def sanitize(raw_input: str) -> str:
    """PHASE 1: Sanitization & Normalization."""
    return raw_input.lower().strip()


def get_response(user_input: str) -> str:
    """PHASE: Process — intent matching via dictionary lookup (O(1))."""
    # Direct exact-match lookup with a graceful fallback (the ".get()" pattern)
    if user_input in responses:
        reply = responses[user_input]
    else:
        # Nested logic: simple partial-match check for a bit of flexibility
        reply = None
        for intent, resp in responses.items():
            if intent in user_input:
                reply = resp
                break
        if reply is None:
            reply = FALLBACK_RESPONSES

    return random.choice(reply) if isinstance(reply, list) else reply


def run_chatbot():
    print(f"🤖 {BOT_NAME}: Hello! I'm your Project 1 rule-based chatbot.")
    print(f"🤖 {BOT_NAME}: Type 'bye' / 'exit' / 'quit' anytime to leave.\n")

    # PHASE: The Heartbeat — infinite loop until the kill command
    while True:
        raw_input_text = input("You: ")
        user_input = sanitize(raw_input_text)

        if not user_input:
            continue

        if user_input in EXIT_COMMANDS:
            print(f"🤖 {BOT_NAME}: Goodbye! Have a great day. 👋")
            break

        reply = get_response(user_input)
        print(f"🤖 {BOT_NAME}: {reply}")


if __name__ == "__main__":
    run_chatbot()
