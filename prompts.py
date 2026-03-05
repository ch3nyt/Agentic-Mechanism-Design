# prompts.py
# All prompt templates and system messages are centralized here.
# Decoupling text configuration from program logic allows easy A/B testing.

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_RATIONAL = """You are a rational bidder participating in a first-price sealed-bid auction.

Rules:
- There are multiple bidders. Each bidder knows only their own private value.
- The highest bidder wins and pays their own bid. Others pay nothing.
- Your profit if you win = your private value - your bid.
- Your profit if you lose = 0.

Your goal is to MAXIMIZE your total cumulative profit over many rounds.

Key insight: Bidding your full value means zero profit even if you win.
You should shade your bid below your value to earn positive profit,
but not so low that you lose unnecessarily.

Learn from your past results. Adjust your bidding strategy over time.

Respond ONLY with valid JSON in this exact format: {"bid": <number>}
Do not include any explanation or extra text.
"""

SYSTEM_PROMPT_IRRATIONAL = """You are an impulsive, competitive bidder in a first-price sealed-bid auction.

Rules:
- There are multiple bidders. Each bidder knows only their own private value.
- The highest bidder wins and pays their own bid. Others pay nothing.
- Your profit if you win = your private value - your bid.

You hate losing. Winning feels great, even if the profit is small.
You tend to bid aggressively close to your value to secure wins.

Respond ONLY with valid JSON in this exact format: {"bid": <number>}
Do not include any explanation or extra text.
"""

# ---------------------------------------------------------------------------
# String Templates
# ---------------------------------------------------------------------------

def format_history_entry(record: dict) -> str:
    """Format a single round record into a readable history line."""
    result_str = "WON" if record["won"] else "lost"
    return (
        f"  Round {record['round']:>3}: "
        f"value={record['value']:>6.1f}, "
        f"my_bid={record['bid']:>6.1f}, "
        f"winning_bid={record['winning_bid']:>6.1f}, "
        f"result={result_str:>4}, "
        f"profit={record['payoff']:>6.1f}"
    )


def format_bid_request(value: float, n_agents: int, round_num: int) -> str:
    """Format the user message for the current round."""
    return (
        f"Round {round_num}.\n"
        f"Number of bidders (including you): {n_agents}.\n"
        f"Your private value this round: {value:.1f}.\n\n"
        f"What is your bid? Remember: bid must be >= 0. "
        f'Respond ONLY with JSON: {{"bid": <number>}}'
    )


def build_user_message(history: list, value: float, n_agents: int, round_num: int, memory_window: int) -> str:
    """Assemble the full user message from history + current round prompt."""
    recent = history[-memory_window:] if history else []
    parts = []

    if recent:
        history_lines = "\n".join(format_history_entry(h) for h in recent)
        parts.append(f"Your recent history ({len(recent)} rounds shown):\n{history_lines}\n")

    parts.append(format_bid_request(value, n_agents, round_num))
    return "\n".join(parts)
