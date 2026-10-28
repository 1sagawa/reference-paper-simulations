"""
bargaining_sim/bayesian_agent.py
論文 Appendix A.3 のアルゴリズムに基づくベイズ型エージェント

核心ロジック:
  提案時: max_{(x,y)} Σ_{v_{-i}} 1{∃j≠i: accept(v_j, x_g, y_r)} × Δu_i × B_i(v_{-i})
  更新時: accept→受け入れ整合的なv_jだけ残す / reject→受け入れる評価額を捨てる
"""
import numpy as np
from itertools import product as iproduct
from game import BargainingGame, Trade, TradeResult, N_PLAYERS, PRIVATE_LOW, PRIVATE_HIGH, GREEN_VALUE


# 信念用の評価額グリッド（離散化）
N_BELIEF_BINS = 20


def make_value_grid(colors: list[str]) -> dict[str, np.ndarray]:
    """非緑色チップの評価額グリッドを作る"""
    grid = {}
    for c in colors:
        if c == "green":
            grid[c] = np.array([GREEN_VALUE])
        else:
            grid[c] = np.linspace(PRIVATE_LOW, PRIVATE_HIGH, N_BELIEF_BINS)
    return grid


class BayesianAgent:
    """
    1プレイヤー分のベイズ型エージェント。
    他の全プレイヤーの各色チップの評価額について一様事前分布を持ち、
    取引結果を観測するたびに信念を更新する。
    """

    def __init__(self, player_id: int, game: BargainingGame):
        self.pid = player_id
        self.game = game
        self.colors = game.colors
        self.value_grid = make_value_grid(self.colors)

        # beliefs[j][color] = 正規化済み確率ベクトル (len = N_BELIEF_BINS)
        # 自分以外のプレイヤーについて一様事前分布で初期化
        self.beliefs: dict[int, dict[str, np.ndarray]] = {}
        for j in range(N_PLAYERS):
            if j == self.pid:
                continue
            self.beliefs[j] = {}
            for c in self.colors:
                grid = self.value_grid[c]
                self.beliefs[j][c] = np.ones(len(grid)) / len(grid)

    # ────────────────────────────────────────
    # 提案生成
    # ────────────────────────────────────────
    def propose(self) -> Trade:
        """期待サープラスを最大化する取引を選ぶ"""
        best_trade = None
        best_ev = -np.inf

        chips_p = self.game.chips[self.pid]
        val_p   = self.game.valuations[self.pid]

        # 全ての (give_color, get_color, give_qty, get_qty) を探索
        for give_c in self.colors:
            for get_c in self.colors:
                if give_c == get_c:
                    continue
                max_give = chips_p[give_c]
                if max_give == 0:
                    continue
                for give_q in range(1, max_give + 1):
                    for get_q in range(1, 11):  # 相手が持てる最大は10
                        # 自分の利得
                        delta_u = val_p[get_c] * get_q - val_p[give_c] * give_q
                        if delta_u <= 0:
                            continue   # 自分が得をしない提案は出さない

                        trade = Trade(
                            proposer_id=self.pid,
                            give_color=give_c,
                            give_qty=give_q,
                            get_color=get_c,
                            get_qty=get_q,
                        )

                        # 期待値 = P(誰か1人以上が受け入れる) × delta_u
                        p_any_accept = self._prob_any_accept(trade)
                        ev = p_any_accept * delta_u

                        if ev > best_ev:
                            best_ev = ev
                            best_trade = trade

        # 良い取引が見つからない場合は最小コストの取引にフォールバック
        if best_trade is None or best_ev <= 0:
            best_trade = self._fallback_trade()

        return best_trade

    def _prob_any_accept(self, trade: Trade) -> float:
        """
        少なくとも1人の相手が受け入れる確率を計算する。
        myopically rational: v_j * get_qty - v_i_give * give_qty > 0
        → receiver にとって: v_j[give_c] * give_qty > v_j[get_c] * get_qty
        """
        give_c = trade.give_color
        get_c  = trade.get_color
        give_q = trade.give_qty
        get_q  = trade.get_qty

        # 各相手が受け入れない確率
        p_none_accept = 1.0
        for j, bel in self.beliefs.items():
            if not self.game.can_accept(j, trade):
                # チップが足りないので必ず拒否
                continue
            # v_j[give_c] * give_q > v_j[get_c] * get_q  →  受け入れ
            grid_give = self.value_grid[give_c]
            grid_get  = self.value_grid[get_c]
            prob_give  = bel[give_c]   # shape (N_BELIEF_BINS,) or (1,)
            prob_get   = bel[get_c]

            # 受け入れ条件を満たす (v_give, v_get) の組み合わせの確率
            p_accept_j = 0.0
            for gi, vg in enumerate(grid_give):
                for ki, vk in enumerate(grid_get):
                    if vg * give_q > vk * get_q:
                        p_accept_j += prob_give[gi] * prob_get[ki]

            p_reject_j = 1.0 - p_accept_j
            p_none_accept *= p_reject_j

        return 1.0 - p_none_accept

    def _fallback_trade(self) -> Trade:
        """有効な提案を1つ強制的に作る（緑チップ1枚 ↔ 他の色1枚）"""
        chips_p = self.game.chips[self.pid]
        for give_c in self.colors:
            if give_c == "green" or chips_p[give_c] == 0:
                continue
            if chips_p["green"] == 0:
                continue
            return Trade(self.pid, "green", 1, give_c, 1)
        # 最後の手段：給付1枚要求
        for c in self.colors:
            if c != "green" and chips_p["green"] > 0:
                return Trade(self.pid, "green", 1, c, 1)
        return Trade(self.pid, self.colors[0], 1, self.colors[1], 1)

    # ────────────────────────────────────────
    # 受諾判断（myopically rational）
    # ────────────────────────────────────────
    def decide_accept(self, trade: Trade) -> bool:
        """自分の評価額に基づいてmyopically rational に判断"""
        val = self.game.valuations[self.pid]
        chips = self.game.chips[self.pid]
        if chips[trade.get_color] < trade.get_qty:
            return False  # チップが足りない
        gain  = val[trade.give_color] * trade.give_qty
        cost  = val[trade.get_color]  * trade.get_qty
        return gain > cost

    # ────────────────────────────────────────
    # 信念更新（Appendix A.3 BayesianUpdate）
    # ────────────────────────────────────────
    def update_beliefs(self, result: TradeResult):
        """
        取引結果を観測して信念を更新する。
        - accepted: 取引相手 r については「受け入れ整合的」な評価額だけ残す
        - rejected (no one): 全員について「拒否整合的」な評価額だけ残す
        """
        trade = result.trade
        give_c = trade.give_color
        get_c  = trade.get_color
        give_q = trade.give_qty
        get_q  = trade.get_qty

        if result.outcome == "accepted":
            r = result.accepted_by
            # r が受け入れた → v_r[give_c] * give_q > v_r[get_c] * get_q
            if r != self.pid and r in self.beliefs:
                self._keep_accept_consistent(r, give_c, give_q, get_c, get_q)

            # 観測された拒否者（受け入れなかった方）についても更新
            for j in self.beliefs:
                if j != r and j != trade.proposer_id:
                    # j は明示的に拒否 → 拒否整合的に更新
                    self._keep_reject_consistent(j, give_c, give_q, get_c, get_q)
        else:
            # 全員拒否
            for j in self.beliefs:
                if j != trade.proposer_id:
                    self._keep_reject_consistent(j, give_c, give_q, get_c, get_q)

    def _keep_accept_consistent(self, j, give_c, give_q, get_c, get_q):
        """j が受け入れと整合的な評価額だけ残す（renormalize）"""
        bel = self.beliefs[j]
        grid_give = self.value_grid[give_c]
        grid_get  = self.value_grid[get_c]

        # v_give[gi] * give_q > v_get[ki] * get_q を満たす (gi, ki) ペアを特定
        # 独立信念として近似：色ごとに独立に処理
        mask_give = (grid_give * give_q)        # shape (G,)
        mask_get  = (grid_get  * get_q)         # shape (K,)

        # 各 give 評価額について、受け入れ可能な get 評価額の期待確率の重み
        new_prob_give = np.zeros_like(bel[give_c])
        new_prob_get  = np.zeros_like(bel[get_c])

        for gi, vg in enumerate(grid_give):
            for ki, vk in enumerate(grid_get):
                if vg * give_q > vk * get_q:
                    new_prob_give[gi] += bel[get_c][ki]
                    new_prob_get[ki]  += bel[give_c][gi]

        # renormalize
        self.beliefs[j][give_c] = self._normalize(new_prob_give * bel[give_c])
        self.beliefs[j][get_c]  = self._normalize(new_prob_get  * bel[get_c])

    def _keep_reject_consistent(self, j, give_c, give_q, get_c, get_q):
        """j が拒否と整合的な評価額だけ残す"""
        bel = self.beliefs[j]
        grid_give = self.value_grid[give_c]
        grid_get  = self.value_grid[get_c]

        new_prob_give = np.zeros_like(bel[give_c])
        new_prob_get  = np.zeros_like(bel[get_c])

        for gi, vg in enumerate(grid_give):
            for ki, vk in enumerate(grid_get):
                if vg * give_q <= vk * get_q:   # 拒否条件
                    new_prob_give[gi] += bel[get_c][ki]
                    new_prob_get[ki]  += bel[give_c][gi]

        s_give = (new_prob_give * bel[give_c]).sum()
        s_get  = (new_prob_get  * bel[get_c]).sum()
        if s_give > 1e-10:
            self.beliefs[j][give_c] = self._normalize(new_prob_give * bel[give_c])
        if s_get > 1e-10:
            self.beliefs[j][get_c]  = self._normalize(new_prob_get  * bel[get_c])

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        s = arr.sum()
        if s < 1e-10:
            return np.ones_like(arr) / len(arr)  # フォールバック: 一様に戻す
        return arr / s
