"""
bargaining_sim/game.py
ゲームの骨格：状態管理・ターン進行・取引処理・サープラス計算
"""
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────
# 定数
# ──────────────────────────────────────────
COLORS_2 = ["green", "red"]
COLORS_3 = ["green", "red", "blue"]
COLORS_4 = ["green", "red", "blue", "purple"]

GREEN_VALUE = 0.50          # 緑チップは全員共通
PRIVATE_LOW  = 0.10
PRIVATE_HIGH = 1.00
CHIPS_PER_COLOR = 10
N_PLAYERS = 3
N_TURNS = 9                 # 全9ターン（1人3回）


# ──────────────────────────────────────────
# データクラス
# ──────────────────────────────────────────
@dataclass
class Trade:
    """1回の取引提案を表す"""
    proposer_id: int
    give_color: str
    give_qty: int
    get_color: str
    get_qty: int

    def __str__(self):
        return (f"P{self.proposer_id+1}: {self.give_qty}×{self.give_color}"
                f" → {self.get_qty}×{self.get_color}")


@dataclass
class TradeResult:
    trade: Trade
    accepted_by: Optional[int]   # None=誰も受けなかった、int=取引相手のID
    outcome: str                 # "accepted" / "rejected"


@dataclass
class GameState:
    """ゲーム全体の状態スナップショット（ターンごとに記録）"""
    turn: int
    chips: list[dict]            # chips[i] = {"green": n, "red": n, ...}
    history: list[TradeResult]
    proposer_id: int


# ──────────────────────────────────────────
# メインのゲームクラス
# ──────────────────────────────────────────
class BargainingGame:
    def __init__(self, colors: list[str] = COLORS_2, seed: int = None):
        self.colors = colors
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # チップ配布
        self.chips: list[dict] = []         # chips[i][color] = count
        self.valuations: list[dict] = []    # valuations[i][color] = $/chip
        self.initial_chips: list[dict] = []
        self._init_chips()

        # ターン管理
        self.turn_order = self._make_turn_order()  # [0,1,2,0,1,2,0,1,2]
        self.current_turn_idx = 0
        self.history: list[TradeResult] = []

    # ── 初期化 ─────────────────────────────
    def _init_chips(self):
        for i in range(N_PLAYERS):
            chips_i = {c: CHIPS_PER_COLOR for c in self.colors}
            val_i = {}
            for c in self.colors:
                if c == "green":
                    val_i[c] = GREEN_VALUE
                else:
                    val_i[c] = round(
                        self.np_rng.uniform(PRIVATE_LOW, PRIVATE_HIGH), 2
                    )
            self.chips.append(chips_i)
            self.valuations.append(val_i)
            self.initial_chips.append(dict(chips_i))

    def _make_turn_order(self) -> list[int]:
        order = list(range(N_PLAYERS)) * 3   # [0,1,2] × 3 = 9ターン
        # 最初だけシャッフルして固定（論文では1ラウンド目にランダム決定）
        first_round = list(range(N_PLAYERS))
        self.rng.shuffle(first_round)
        return first_round * 3

    # ── ゲーム情報取得 ─────────────────────
    @property
    def n_turns_total(self) -> int:
        return N_TURNS

    @property
    def current_proposer(self) -> int:
        return self.turn_order[self.current_turn_idx]

    @property
    def is_done(self) -> bool:
        return self.current_turn_idx >= N_TURNS

    def get_state(self) -> GameState:
        return GameState(
            turn=self.current_turn_idx,
            chips=[dict(c) for c in self.chips],
            history=list(self.history),
            proposer_id=self.current_proposer,
        )

    def welfare(self, player_id: int, chips: dict = None) -> float:
        """指定チップセットでの総価値を返す（省略時は現在の手持ち）"""
        c = chips if chips is not None else self.chips[player_id]
        v = self.valuations[player_id]
        return sum(c[color] * v[color] for color in self.colors)

    def initial_welfare(self, player_id: int) -> float:
        return self.welfare(player_id, self.initial_chips[player_id])

    def surplus(self, player_id: int) -> float:
        return max(0.0, self.welfare(player_id) - self.initial_welfare(player_id))

    def total_surplus(self) -> float:
        return sum(self.surplus(i) for i in range(N_PLAYERS))

    # ── バリデーション ─────────────────────
    def is_valid_trade(self, trade: Trade) -> bool:
        """提案が物理的に有効かチェック"""
        p = trade.proposer_id
        if trade.give_color == trade.get_color:
            return False
        if trade.give_color not in self.colors or trade.get_color not in self.colors:
            return False
        if trade.give_qty <= 0 or trade.get_qty <= 0:
            return False
        if self.chips[p][trade.give_color] < trade.give_qty:
            return False
        return True

    def can_accept(self, receiver_id: int, trade: Trade) -> bool:
        """受け手が受諾できるか（チップが足りるか）"""
        return self.chips[receiver_id][trade.get_color] >= trade.get_qty

    # ── 取引実行 ───────────────────────────
    def execute_turn(self, trade: Trade, responses: dict[int, bool]) -> TradeResult:
        """
        1ターンを実行する。
        responses: {player_id: True/False} (提案者以外の2人分)
        """
        assert self.current_turn_idx < N_TURNS
        assert trade.proposer_id == self.current_proposer

        # 実際に受け入れ可能なプレイヤーだけ有効
        valid_acceptors = [
            pid for pid, accepted in responses.items()
            if accepted and self.can_accept(pid, trade)
        ]

        result: TradeResult
        if valid_acceptors:
            chosen = self.rng.choice(valid_acceptors)   # ランダムに1人選ぶ
            p = trade.proposer_id
            r = chosen
            # チップ交換
            self.chips[p][trade.give_color] -= trade.give_qty
            self.chips[p][trade.get_color]  += trade.get_qty
            self.chips[r][trade.get_color]  -= trade.get_qty
            self.chips[r][trade.give_color] += trade.give_qty
            result = TradeResult(trade=trade, accepted_by=chosen, outcome="accepted")
        else:
            result = TradeResult(trade=trade, accepted_by=None, outcome="rejected")

        self.history.append(result)
        self.current_turn_idx += 1
        return result

    # ── 表示用ユーティリティ ───────────────
    def format_history(self, n_last: int = None) -> str:
        items = self.history if n_last is None else self.history[-n_last:]
        lines = []
        for tr in items:
            t = tr.trade
            if tr.outcome == "accepted":
                lines.append(
                    f"  Turn{self.history.index(tr)+1}: {t} → accepted by P{tr.accepted_by+1}"
                )
            else:
                lines.append(f"  Turn{self.history.index(tr)+1}: {t} → rejected")
        return "\n".join(lines) if lines else "  (なし)"

    def print_status(self):
        print(f"\n=== Turn {self.current_turn_idx}/{N_TURNS} ===")
        header = f"{'Player':<8}" + "".join(f"{c:>8}" for c in self.colors) + f"{'Welfare':>10}{'Surplus':>10}"
        print(header)
        for i in range(N_PLAYERS):
            row = f"  P{i+1}   "
            row += "".join(f"{self.chips[i][c]:>8}" for c in self.colors)
            row += f"{self.welfare(i):>10.2f}{self.surplus(i):>10.2f}"
            print(row)
        print(f"  Total surplus: {self.total_surplus():.2f}")
