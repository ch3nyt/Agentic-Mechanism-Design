# main.py
# Central orchestrator for the multi-agent auction experiment.
# Instantiates environment and agents, runs N_ROUNDS, logs results to CSV.

import csv
import os
import time

from dotenv import load_dotenv

from agent import AuctionAgent
from environment import AuctionEnvironment

load_dotenv("api.env")

# ---------------------------------------------------------------------------
# Experiment configuration (override via api.env)
# ---------------------------------------------------------------------------
N_AGENTS = int(os.getenv("N_AGENTS", 3))
N_ROUNDS = int(os.getenv("N_ROUNDS", 50))
VALUE_LOW = float(os.getenv("VALUE_LOW", 0))
VALUE_HIGH = float(os.getenv("VALUE_HIGH", 100))
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", 10))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
RESULTS_FILE = os.getenv("RESULTS_FILE", "results.csv")

# Assign personas: first agent is rational, rest are rational by default.
# Change this list to mix rational / irrational agents for A/B experiments.
AGENT_PERSONAS = ["rational"] * N_AGENTS
# Example: AGENT_PERSONAS = ["rational", "irrational", "rational"]

# Delay between rounds (seconds) to avoid rate-limit errors on free-tier APIs.
ROUND_DELAY = 0.5

CSV_FIELDS = [
    "round",
    "agent_id",
    "persona",
    "value",
    "bid",
    "bid_ratio",      # bid / value — useful for convergence analysis
    "bne_bid",        # theoretical BNE bid for reference
    "won",
    "payoff",
    "winning_bid",
    "cumulative_profit",
]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print(f"=== Auction Experiment ===")
    print(f"Agents: {N_AGENTS} | Rounds: {N_ROUNDS} | Value range: [{VALUE_LOW}, {VALUE_HIGH}]")
    print(f"Personas: {AGENT_PERSONAS}")
    print(f"Results -> {RESULTS_FILE}\n")

    env = AuctionEnvironment(
        n_agents=N_AGENTS,
        value_low=VALUE_LOW,
        value_high=VALUE_HIGH,
    )

    agents = [
        AuctionAgent(
            agent_id=i,
            n_agents=N_AGENTS,
            persona=AGENT_PERSONAS[i],
            memory_window=MEMORY_WINDOW,
            temperature=TEMPERATURE,
        )
        for i in range(N_AGENTS)
    ]

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for round_num in range(1, N_ROUNDS + 1):
            print(f"--- Round {round_num:>3}/{N_ROUNDS} ---")

            # Step 1: generate private values
            values = env.generate_values()

            # Step 2: collect bids from all agents
            bids = []
            for agent in agents:
                bid = agent.decide_bid(values[agent.agent_id], round_num)
                bids.append(bid)
                print(
                    f"  Agent {agent.agent_id} [{agent.persona:>10}]: "
                    f"value={values[agent.agent_id]:>6.1f}  bid={bid:>6.1f}"
                )

            # Step 3: settle the auction
            result = env.resolve(values, bids)
            winner_id = result["winner_id"]
            winning_bid = result["winning_bid"]
            print(
                f"  => Winner: Agent {winner_id} "
                f"(bid={winning_bid:.1f}, "
                f"profit={result['payoffs'][winner_id]:.1f})\n"
            )

            # Step 4: update memory and write CSV row for each agent
            for agent in agents:
                i = agent.agent_id
                won = i == winner_id
                payoff = result["payoffs"][i]

                agent.update_memory(
                    value=values[i],
                    bid=bids[i],
                    won=won,
                    payoff=payoff,
                    winning_bid=winning_bid,
                    round_num=round_num,
                )

                bid_ratio = round(bids[i] / values[i], 4) if values[i] > 0 else None
                bne = round(env.bne_bid(values[i]), 4)

                writer.writerow(
                    {
                        "round": round_num,
                        "agent_id": i,
                        "persona": agent.persona,
                        "value": round(values[i], 2),
                        "bid": round(bids[i], 2),
                        "bid_ratio": bid_ratio,
                        "bne_bid": bne,
                        "won": int(won),
                        "payoff": round(payoff, 2),
                        "winning_bid": round(winning_bid, 2),
                        "cumulative_profit": round(agent.total_profit, 2),
                    }
                )

            # Flush after each round so partial results are saved on interruption
            csvfile.flush()

            if round_num < N_ROUNDS:
                time.sleep(ROUND_DELAY)

    # Final summary
    print("=== Experiment Complete ===")
    for agent in agents:
        print(
            f"  Agent {agent.agent_id} [{agent.persona}]: "
            f"total_profit={agent.total_profit:.1f}  "
            f"win_rate={agent.win_rate:.1%}"
        )
    print(f"\nRaw data saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
