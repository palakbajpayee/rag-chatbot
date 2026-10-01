"""The customer-service agent: Claude + retrieval-augmented tools in a tool-use loop."""
from dataclasses import dataclass, field

import anthropic

import config
from tools import TOOLS, ToolBox

SYSTEM_PROMPT = f"""You are the customer-service assistant for {config.STORE_NAME}, an online store.

How to answer:
- For any question about store policies or procedures (shipping, returns, refunds, payments, \
accounts, warranty), call search_knowledge_base first and base your answer only on what it returns. \
Cite the article you used in square brackets, e.g. [returns_refunds.md]. If the knowledge base \
doesn't cover it, say so and offer to open a support ticket. Never invent policies, prices, dates \
or order details.
- For questions about a specific order, you need the order id and the email on the order before \
calling lookup_order. Ask for whichever is missing. Never reveal details of an order the lookup \
did not return.
- For product availability, price or warranty, use search_products.
- Open a support ticket with create_support_ticket when the customer asks for a human, when the \
issue needs a policy exception, or involves damaged/missing items or payment disputes. Tell the \
customer the ticket id.

Style: friendly, concise and practical, in short paragraphs or brief steps. Respond in the \
customer's language. Stay on topic: politely decline requests unrelated to shopping at {config.STORE_NAME}."""

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Reply:
    text: str
    tool_calls: list[dict] = field(default_factory=list)  # [{"name", "input", "is_error"}] for debugging/UI


class SupportBot:
    def __init__(self, client: anthropic.Anthropic | None = None, toolbox: ToolBox | None = None):
        self.client = client or anthropic.Anthropic()
        self.toolbox = toolbox or ToolBox()
        self.messages: list[dict] = []

    def reset(self) -> None:
        self.messages = []

    def chat(self, user_text: str) -> Reply:
        turn_start = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})
        tool_calls: list[dict] = []

        try:
            for _ in range(config.MAX_TOOL_ROUNDS):
                response = self.client.beta.messages.create(
                    model=config.MODEL,
                    max_tokens=config.MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=self.messages,
                    output_config={"effort": config.EFFORT},
                    cache_control={"type": "ephemeral"},  # caches system + tools + history between turns
                    betas=[FALLBACK_BETA],
                    fallbacks="default",  # if a safety classifier declines, the API retries on a fallback model
                )

                if response.stop_reason == "refusal":
                    # Drop this whole turn so the conversation stays valid and the customer can rephrase.
                    del self.messages[turn_start:]
                    return Reply("Sorry, I can't help with that request. Is there anything else about "
                                 "your order or our store I can help with?", tool_calls)

                # Append the full content (not just the text) so thinking/tool blocks are preserved.
                self.messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "tool_use":
                    results = []
                    for block in response.content:
                        if block.type != "tool_use":
                            continue
                        output, is_error = self.toolbox.run(block.name, dict(block.input))
                        tool_calls.append({"name": block.name, "input": dict(block.input), "is_error": is_error})
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": output, "is_error": is_error})
                    # All results for one assistant turn go back in a single user message.
                    self.messages.append({"role": "user", "content": results})
                    continue

                if response.stop_reason == "pause_turn":
                    continue  # the server paused a long turn; resend to let it finish

                text = "".join(b.text for b in response.content if b.type == "text").strip()
                if response.stop_reason == "max_tokens":
                    text += "\n\n(Response was cut short - ask me to continue.)"
                return Reply(text or "Sorry, I didn't catch that. Could you rephrase?", tool_calls)
        except anthropic.APIError:
            # Roll back the half-finished turn so the next message starts from a valid history.
            del self.messages[turn_start:]
            raise

        del self.messages[turn_start:]
        return Reply("Sorry, I couldn't finish that request. I can connect you with a human agent - "
                     "could you share your email so I can open a support ticket?", tool_calls)
