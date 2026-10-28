"""
bargaining_sim/agents.py
人間エージェント（ルールベース）とLLMエージェント（GPT-4o）
"""
import os
import re
import random
from game import BargainingGame, Trade, N_PLAYERS


# ════════════════════════════════════════════
# 人間エージェント（ルールベース）
# ════════════════════════════════════════════
class HumanAgent:
    """
    論文の観察に基づく人間エージェント:
    - 公平な 1:1 交換（チップ数が等しい取引）を好む
    - 自分が利益を得る取引を優先するが、相手にも得になるよう配慮する
    - 価値の低い色を手放して高い色を得ようとする
    """

    def __init__(self, player_id: int, game: BargainingGame, seed: int = None):
        self.pid = player_id
        self.game = game
        self.rng = random.Random(seed)

    def propose(self) -> Trade:
        val = self.game.valuations[self.pid]
        chips = self.game.chips[self.pid]
        colors = self.game.colors

        # 自分の評価が低い色 → 高い色 を探す（1:1交換）
        best_trade = None
        best_gain  = -float("inf")

        for give_c in colors:
            if chips[give_c] == 0:
                continue
            for get_c in colors:
                if give_c == get_c:
                    continue
                if val[give_c] >= val[get_c]:
                    continue  # 価値が上がらない交換は無視

                # 1:1 交換（公平性規範）
                qty = min(chips[give_c], 5)          # 一度に出しすぎない
                qty = max(1, self.rng.randint(1, qty))
                gain = (val[get_c] - val[give_c]) * qty

                if gain > best_gain:
                    best_gain  = gain
                    best_trade = Trade(self.pid, give_c, qty, get_c, qty)

        # 良い取引がなければフォールバック（最もvalueが低い色を渡す）
        if best_trade is None:
            give_c = min([c for c in colors if chips[c] > 0],
                         key=lambda c: val[c])
            get_c  = max([c for c in colors if c != give_c],
                         key=lambda c: val[c])
            qty    = max(1, self.rng.randint(1, min(chips[give_c], 3)))
            best_trade = Trade(self.pid, give_c, qty, get_c, qty)

        return best_trade

    def decide_accept(self, trade: Trade) -> bool:
        val   = self.game.valuations[self.pid]
        chips = self.game.chips[self.pid]
        if chips[trade.get_color] < trade.get_qty:
            return False
        gain = val[trade.give_color] * trade.give_qty - val[trade.get_color] * trade.get_qty
        # 自分が利益を得る場合は受け入れ。わずかなマイナスでも公平性のため受け入れることも
        return gain >= -0.05


# ════════════════════════════════════════════
# LLMエージェント（GPT-4o）
# ════════════════════════════════════════════
class LLMAgent:
    """
    論文 Appendix H.2 / H.3 のプロンプトに基づく GPT-4o エージェント。
    OPENAI_API_KEY 環境変数が設定されている必要がある。
    """

    def __init__(self, player_id: int, game: BargainingGame,
                 name: str = None, model: str = "gpt-4o"):
        self.pid   = player_id
        self.game  = game
        self.model = model
        self.name  = name or f"Player{player_id + 1}"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except ImportError:
                raise RuntimeError("openai パッケージが必要です: pip install openai")
        return self._client

    # ── 情報フォーマット ───────────────────
    def _pref_desc(self) -> str:
        val = self.game.valuations[self.pid]
        parts = [f"{c}: ${val[c]:.2f}/chip" for c in self.game.colors]
        return ", ".join(parts)

    def _item_desc(self) -> str:
        chips = self.game.chips[self.pid]
        parts = [f"{c}: {chips[c]}" for c in self.game.colors]
        return ", ".join(parts)

    def _history_desc(self) -> str:
        if not self.game.history:
            return "No trades yet."
        lines = []
        for i, tr in enumerate(self.game.history):
            t = tr.trade
            if tr.outcome == "accepted":
                lines.append(
                    f"Turn {i+1}: P{t.proposer_id+1} offered {t.give_qty} {t.give_color} "
                    f"for {t.get_qty} {t.get_color} → accepted by P{tr.accepted_by+1}"
                )
            else:
                lines.append(
                    f"Turn {i+1}: P{t.proposer_id+1} offered {t.give_qty} {t.give_color} "
                    f"for {t.get_qty} {t.get_color} → rejected"
                )
        return "\n".join(lines)

    def _all_chips_desc(self) -> str:
        lines = []
        for i in range(N_PLAYERS):
            parts = [f"{c}: {self.game.chips[i][c]}" for c in self.game.colors]
            lines.append(f"  P{i+1}: " + ", ".join(parts))
        return "\n".join(lines)

    # ── 提案プロンプト (Appendix H.2) ──────
    def propose(self) -> Trade:
        prompt = f"""You are {self.name}.
Your valuations of the different types of chips are: {self._pref_desc()}.
You now have the following amounts of each chip: {self._item_desc()}.
All players' chip counts:
{self._all_chips_desc()}
The conversation history so far is:
{self._history_desc()}

REMEMBER, to propose a trade, you must:
Request a certain quantity of chips of a single color from any other player to get
Specify a certain quantity of chips of a different color to give in return

REMEMBER you have the following amounts of each chip: {self._item_desc()}.
Your goal is to make as much money as possible. The trades you choose to make to accomplish this are up to you.
As a part of making money you must be rational - do not propose a trade in which you lose money.
The value of a trade to you is the difference between the total value of chips you receive (quantity x your valuation) minus the total value of chips you give up (quantity x your valuation).
Only propose trades that give you positive value.
In short, your trades should be both incentive compatible and individually rational.

Your response must use these EXACT tags below. The response should include nothing else besides the tags, your trade offer, and your reasoning.
---
<REASONING>
[Provide your concise reasoning in a few sentences]
</REASONING>
<CHECK>
[check if you have sufficient chips to trade]
</CHECK>
<GET_COLOR> Color, e.g. red</GET_COLOR>
<GET_QUANTITY> quantity, e.g. n </GET_QUANTITY>
<GIVE_COLOR> Color, e.g. red</GIVE_COLOR>
<GIVE_QUANTITY> quantity, e.g. n </GIVE_QUANTITY>
---"""

        response_text = self._call_api(prompt)
        trade = self._parse_proposal(response_text)
        if trade is None or not self.game.is_valid_trade(trade):
            trade = self._fallback_trade()
        return trade

    # ── 受諾判断プロンプト (Appendix H.3) ──
    def decide_accept(self, trade: Trade) -> bool:
        val = self.game.valuations[self.pid]
        chips = self.game.chips[self.pid]
        if chips[trade.get_color] < trade.get_qty:
            return False

        delta = val[trade.give_color] * trade.give_qty - val[trade.get_color] * trade.get_qty

        prompt = f"""You are {self.name}.
Your valuations of the different types of chips are: {self._pref_desc()}.
You now have the following amounts of each chip: {self._item_desc()}.
The conversation history so far is:
{self._history_desc()}

You have an offer. P{trade.proposer_id+1} is offering to give {trade.give_qty} {trade.give_color} and get {trade.get_qty} {trade.get_color} in return.
If you make this trade, your total wealth will change by: ${delta:+.2f}

Now, you need to decide whether to accept or decline.
Your response must use these EXACT tags below.
---
<REASONING>
[Provide your concise reasoning in a few sentences.]
</REASONING>
<CHOICE>Yes or No</CHOICE>
---"""

        response_text = self._call_api(prompt)
        return self._parse_acceptance(response_text)

    # ── API呼び出し ────────────────────────
    def _call_api(self, prompt: str) -> str:
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"<CHOICE>No</CHOICE>  <!-- API error: {e} -->"

    # ── パース ─────────────────────────────
    def _parse_proposal(self, text: str) -> Trade | None:
        try:
            get_color = re.search(r"<GET_COLOR>\s*(\w+)\s*</GET_COLOR>", text, re.I)
            get_qty   = re.search(r"<GET_QUANTITY>\s*(\d+)\s*</GET_QUANTITY>", text, re.I)
            give_color= re.search(r"<GIVE_COLOR>\s*(\w+)\s*</GIVE_COLOR>", text, re.I)
            give_qty  = re.search(r"<GIVE_QUANTITY>\s*(\d+)\s*</GIVE_QUANTITY>", text, re.I)
            if not all([get_color, get_qty, give_color, give_qty]):
                return None
            return Trade(
                proposer_id=self.pid,
                give_color=give_color.group(1).lower(),
                give_qty=int(give_qty.group(1)),
                get_color=get_color.group(1).lower(),
                get_qty=int(get_qty.group(1)),
            )
        except Exception:
            return None

    def _parse_acceptance(self, text: str) -> bool:
        match = re.search(r"<CHOICE>\s*(Yes|No)\s*</CHOICE>", text, re.I)
        if match:
            return match.group(1).strip().lower() == "yes"
        # タグがなければ本文中の yes/no を探す
        return "yes" in text.lower()

    def _fallback_trade(self) -> Trade:
        """パース失敗時のフォールバック"""
        val   = self.game.valuations[self.pid]
        chips = self.game.chips[self.pid]
        colors = self.game.colors
        give_c = min([c for c in colors if chips[c] > 0], key=lambda c: val[c])
        get_c  = max([c for c in colors if c != give_c], key=lambda c: val[c])
        return Trade(self.pid, give_c, 1, get_c, 1)
