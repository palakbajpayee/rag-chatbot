"""Terminal chat:  python cli.py   (add --debug to see tool calls)."""
import sys

import anthropic

import config
from chatbot import SupportBot


def main() -> None:
    debug = "--debug" in sys.argv
    bot = SupportBot()
    print(f"{config.STORE_NAME} Support - type 'quit' to exit, 'reset' to start over.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            break
        if user.lower() == "reset":
            bot.reset()
            print("(conversation cleared)\n")
            continue
        try:
            reply = bot.chat(user)
        except anthropic.AuthenticationError:
            sys.exit("Invalid or missing API key. Set ANTHROPIC_API_KEY and try again.")
        except anthropic.RateLimitError:
            print("Assistant: We're busy right now - please try again in a moment.\n")
            continue
        except anthropic.APIStatusError as exc:
            print(f"Assistant: Something went wrong on our side ({exc.status_code}). Please try again.\n")
            continue
        except anthropic.APIConnectionError:
            print("Assistant: I couldn't reach the server. Check your internet connection.\n")
            continue
        if debug:
            for call in reply.tool_calls:
                print(f"  [tool] {call['name']}({call['input']}){' ERROR' if call['is_error'] else ''}")
        print(f"Assistant: {reply.text}\n")


if __name__ == "__main__":
    main()
