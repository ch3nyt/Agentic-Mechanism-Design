# agent.py
# Encapsulates LLM API calls and the In-Context RL memory buffer.
# Each AuctionAgent is an independent entity with its own history.

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from prompts import (
    SYSTEM_PROMPT_IRRATIONAL,
    SYSTEM_PROMPT_RATIONAL,
    build_user_message,
)

load_dotenv("api.env")

# ---------------------------------------------------------------------------
# Persona registry — add new personas here, then pass persona= to AuctionAgent
# ---------------------------------------------------------------------------
PERSONAS = {
    "rational": SYSTEM_PROMPT_RATIONAL,
    "irrational": SYSTEM_PROMPT_IRRATIONAL,
}


class AuctionAgent:
    """
    An LLM-powered bidder that uses its round history as in-context memory.

    Memory buffer stores dicts with keys:
        round, value, bid, won, payoff, winning_bid

    On each round:
        1. decide_bid(value, round_num) -> float
        2. update_memory(...)           -> None   (called by main.py after settlement)
    """

    def __init__(
        self,
        agent_id: int,
        n_agents: int,
        persona: str = "rational",
        memory_window: int = 10,
        temperature: float = 0.7,
    ):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.persona = persona
        self.memory_window = memory_window
        self.temperature = temperature
        self.history: list[dict] = []

        self._system_prompt = PERSONAS.get(persona, SYSTEM_PROMPT_RATIONAL)
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide_bid(self, value: float, round_num: int) -> float:
        """
        Ask the LLM for a bid given the current private value and history.
        Falls back to (value / 2) if the API call or parsing fails.
        """
        messages = self._build_messages(value, round_num)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=64,
            )
            raw_text = response.choices[0].message.content or ""
            bid = self._parse_bid(raw_text, value)
        except Exception as exc:
            print(
                f"  [Agent {self.agent_id}] API error ({type(exc).__name__}: {exc}). "
                f"Falling back to value/2."
            )
            bid = value / 2.0

        return bid

    def update_memory(
        self,
        value: float,
        bid: float,
        won: bool,
        payoff: float,
        winning_bid: float,
        round_num: int,
    ) -> None:
        """Append the round result to this agent's history buffer."""
        self.history.append(
            {
                "round": round_num,
                "value": value,
                "bid": bid,
                "won": won,
                "payoff": payoff,
                "winning_bid": winning_bid,
            }
        )

    @property
    def total_profit(self) -> float:
        return sum(h["payoff"] for h in self.history)

    @property
    def win_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(1 for h in self.history if h["won"]) / len(self.history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(self, value: float, round_num: int) -> list[dict]:
        user_content = build_user_message(
            history=self.history,
            value=value,
            n_agents=self.n_agents,
            round_num=round_num,
            memory_window=self.memory_window,
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _parse_bid(self, text: str, value: float) -> float:
        """
        Robustly extract a float bid from LLM output.
        Priority: JSON parse -> regex number extraction -> fallback.
        Always clamps result to [0, value].
        """
        # 1. Try strict JSON
        try:
            data = json.loads(text.strip())
            bid = float(data["bid"])
            return self._clamp(bid, value)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

        # 2. Try JSON embedded inside prose  {"bid": 42.5}
        json_match = re.search(r'\{[^}]*"bid"\s*:\s*([\d.]+)[^}]*\}', text)
        if json_match:
            try:
                bid = float(json_match.group(1))
                return self._clamp(bid, value)
            except ValueError:
                pass

        # 3. Extract first decimal/integer number from text
        numbers = re.findall(r"\d+\.?\d*", text)
        if numbers:
            bid = float(numbers[0])
            return self._clamp(bid, value)

        # 4. Last resort
        print(
            f"  [Agent {self.agent_id}] Could not parse bid from: {text!r}. "
            f"Falling back to value/2."
        )
        return self._clamp(value / 2.0, value)

    @staticmethod
    def _clamp(bid: float, value: float) -> float:
        """Ensure bid is in [0, value]. Bids above value are economically irrational."""
        return round(max(0.0, min(bid, value)), 4)
