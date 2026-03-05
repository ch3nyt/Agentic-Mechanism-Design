# environment.py
# The auction mechanism and settlement engine.
# Acts as the impartial referee: generates values, collects bids, settles outcomes.

import random


class AuctionEnvironment:
    """
    First-price sealed-bid auction environment.

    Each round:
      1. generate_values()  -> list of N private values, one per agent
      2. resolve(values, bids) -> settlement result dict

    Supports any number of agents and configurable value distribution.
    """

    def __init__(
        self,
        n_agents: int,
        value_low: float = 0.0,
        value_high: float = 100.0,
        seed: int | None = None,
    ):
        self.n_agents = n_agents
        self.value_low = value_low
        self.value_high = value_high
        self._rng = random.Random(seed)
        self.round_num = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_values(self) -> list[float]:
        """Draw N private values iid from U(value_low, value_high)."""
        return [
            round(self._rng.uniform(self.value_low, self.value_high), 4)
            for _ in range(self.n_agents)
        ]

    def resolve(self, values: list[float], bids: list[float]) -> dict:
        """
        Settle the auction given private values and submitted bids.

        Args:
            values: Private values for each agent (index = agent_id).
            bids:   Submitted bids for each agent (index = agent_id).

        Returns:
            result dict with keys:
                round        (int)
                winner_id    (int)   — agent index of the winner
                winning_bid  (float) — highest bid submitted
                values       (list[float])
                bids         (list[float])
                payoffs      (list[float]) — profit per agent; winner gets v - b, rest 0
        """
        if len(values) != self.n_agents or len(bids) != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} values and bids, "
                f"got {len(values)} and {len(bids)}."
            )

        self.round_num += 1

        winning_bid = max(bids)

        # Resolve ties randomly (fair lottery)
        candidates = [i for i, b in enumerate(bids) if b == winning_bid]
        winner_id = self._rng.choice(candidates)

        payoffs = [0.0] * self.n_agents
        payoffs[winner_id] = round(values[winner_id] - bids[winner_id], 4)

        return {
            "round": self.round_num,
            "winner_id": winner_id,
            "winning_bid": winning_bid,
            "values": values,
            "bids": bids,
            "payoffs": payoffs,
        }

    # ------------------------------------------------------------------
    # Game-theory helpers
    # ------------------------------------------------------------------

    def bne_bid(self, value: float) -> float:
        """
        Theoretical Bayesian Nash Equilibrium bid for a symmetric agent
        in a first-price auction with N bidders and U(0, value_high):
            b*(v) = (N-1)/N * v
        """
        return (self.n_agents - 1) / self.n_agents * value
