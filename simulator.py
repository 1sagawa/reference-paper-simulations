"""
bargaining_sim/simulator.py
ゲームループ：任意のエージェントの組み合わせでゲームを実行する
"""
import time
from game import BargainingGame, Trade, N_PLAYERS, COLORS_2, COLORS_3, COLORS_4
from agents import HumanAgent, LLMAgent
from bayesian_agent import BayesianAgent


# ════════════════════════════════════════════
# エージェントファクトリ
# ════════════════════════════════════════════
AGENT_NAMES = ["Alice", "Bob", "Carol"]

def build_agents(agent_types: list[str], game: BargainingGame, seed: int = None):
    """
    agent_types: ["human", "llm", "bayesian"] の組み合わせ (長さ3)
    """
    agents = []
    for i, t in enumerate(agent_types):
        t = t.lower()
        if t == "human":
            agents.append(HumanAgent(i, game, seed=seed))
        elif t == "llm":
            agents.append(LLMAgent(i, game, name=AGENT_NAMES[i]))
        elif t == "bayesian":
            agents.append(BayesianAgent(i, game))
        else:
            raise ValueError(f"未知のエージェントタイプ: {t}")
    return agents


# ════════════════════════════════════════════
# ゲームランナー
# ════════════════════════════════════════════
def run_game(
    agent_types: list[str],
    colors: list[str] = COLORS_2,
    seed: int = None,
    verbose: bool = True,
) -> dict:
    """
    1ゲームを実行して結果を返す。

    Returns:
        {
            "surpluses": [float, float, float],
            "total_surplus": float,
            "trade_count": int,
            "accepted_count": int,
            "history": [...],
        }
    """
    game   = BargainingGame(colors=colors, seed=seed)
    agents = build_agents(agent_types, game, seed=seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"ゲーム開始: {' vs '.join(agent_types)} | 色: {colors}")
        print(f"{'='*60}")
        print("\n【初期チップ・評価額】")
        for i in range(N_PLAYERS):
            val_str = ", ".join(f"{c}=${game.valuations[i][c]:.2f}" for c in colors)
            chip_str = ", ".join(f"{c}:{game.chips[i][c]}" for c in colors)
            print(f"  P{i+1} ({agent_types[i]:8s}): chips=[{chip_str}]  val=[{val_str}]")

    # Bayesian agents 用の信念管理
    bayesian_agents = {i: a for i, a in enumerate(agents) if isinstance(a, BayesianAgent)}

    # ─── ゲームループ ───────────────────────
    for turn_idx in range(N_PLAYERS * 3):
        proposer_id = game.current_proposer
        proposer    = agents[proposer_id]

        if verbose:
            print(f"\n--- Turn {turn_idx+1}/9 | 提案者: P{proposer_id+1} ({agent_types[proposer_id]}) ---")

        # 提案を生成
        trade = proposer.propose()

        # バリデーション：無効な提案はスキップ
        if not game.is_valid_trade(trade):
            if verbose:
                print(f"  ⚠ 無効な提案 ({trade}) → スキップ")
            # ダミーの拒否を記録してターンを進める
            from game import TradeResult
            dummy = TradeResult(trade=trade, accepted_by=None, outcome="rejected")
            game.history.append(dummy)
            game.current_turn_idx += 1
            for ba in bayesian_agents.values():
                ba.update_beliefs(dummy)
            continue

        if verbose:
            print(f"  提案: {trade}")

        # 他の2人が受諾判断
        responses = {}
        for j, agent in enumerate(agents):
            if j == proposer_id:
                continue
            decision = agent.decide_accept(trade)
            responses[j] = decision
            if verbose:
                print(f"    P{j+1} ({agent_types[j]:8s}): {'✓ 受諾' if decision else '✗ 拒否'}")

        # 取引を実行
        result = game.execute_turn(trade, responses)

        if verbose:
            if result.outcome == "accepted":
                print(f"  → 成立！P{result.accepted_by+1} と取引")
            else:
                print(f"  → 不成立")

        # Bayesian agents の信念を更新
        for ba in bayesian_agents.values():
            ba.update_beliefs(result)

    # ─── 結果集計 ───────────────────────────
    surpluses = [game.surplus(i) for i in range(N_PLAYERS)]
    total     = game.total_surplus()
    accepted  = sum(1 for r in game.history if r.outcome == "accepted")

    if verbose:
        print(f"\n{'='*60}")
        print("【ゲーム終了】")
        game.print_status()
        print(f"\n  取引成立: {accepted}/{N_PLAYERS*3} ターン")
        print(f"  個別サープラス: " +
              ", ".join(f"P{i+1}={s:.2f}" for i, s in enumerate(surpluses)))
        print(f"  合計サープラス: {total:.2f}")

    return {
        "surpluses": surpluses,
        "total_surplus": total,
        "trade_count": N_PLAYERS * 3,
        "accepted_count": accepted,
        "history": game.history,
        "game": game,
    }


# ════════════════════════════════════════════
# 複数ゲーム実行 & 集計
# ════════════════════════════════════════════
def run_experiments(
    agent_types: list[str],
    colors: list[str] = COLORS_2,
    n_games: int = 5,
    seed_start: int = 0,
    verbose_each: bool = False,
) -> dict:
    """
    同じエージェント構成で n_games ゲームを回して統計を返す。
    """
    import numpy as np

    all_total    = []
    all_accepted = []
    all_surpluses = [[] for _ in range(N_PLAYERS)]

    for g in range(n_games):
        print(f"\n[ゲーム {g+1}/{n_games}] seed={seed_start+g}")
        res = run_game(agent_types, colors=colors, seed=seed_start+g, verbose=verbose_each)
        all_total.append(res["total_surplus"])
        all_accepted.append(res["accepted_count"])
        for i in range(N_PLAYERS):
            all_surpluses[i].append(res["surpluses"][i])

        print(f"  合計サープラス: {res['total_surplus']:.2f}  "
              f"成立数: {res['accepted_count']}/{res['trade_count']}")

    summary = {
        "mean_total_surplus":  float(np.mean(all_total)),
        "std_total_surplus":   float(np.std(all_total)),
        "mean_accepted_rate":  float(np.mean(all_accepted)) / (N_PLAYERS * 3),
        "per_player_mean":     [float(np.mean(all_surpluses[i])) for i in range(N_PLAYERS)],
        "agent_types":         agent_types,
        "colors":              colors,
        "n_games":             n_games,
    }

    print(f"\n{'='*60}")
    print(f"【{n_games}ゲーム統計】{' vs '.join(agent_types)} | 色: {colors}")
    print(f"  平均合計サープラス : {summary['mean_total_surplus']:.3f} ± {summary['std_total_surplus']:.3f}")
    print(f"  平均取引成立率     : {summary['mean_accepted_rate']:.1%}")
    print(f"  各プレイヤー平均   : " +
          ", ".join(f"P{i+1}={v:.3f}" for i, v in enumerate(summary["per_player_mean"])))

    return summary
