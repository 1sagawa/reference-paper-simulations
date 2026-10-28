"""
多人数・多争点交渉ゲームのシミュレーション
論文：LLM backed autonomous agents in multi-issue multilateral negotiations

使い方:
  python negotiation_game.py [オプション]

オプション:
  --rule      majority / unanimous          （デフォルト: majority）
  --goal      none / joint_welfare / envy_index / personal_score  （デフォルト: joint_welfare）
  --rounds    ラウンド上限の整数              （デフォルト: 20）
  --agents    エージェント数 3〜7             （デフォルト: 3）
  --model     OpenAIのモデル名               （デフォルト: gpt-4o-mini）
  --out       出力ファイル名（例: test3.txt）  （デフォルト: 自動採番）

例:
  python negotiation_game.py --rule majority --goal envy_index --rounds 20 --agents 3
  python negotiation_game.py --rule unanimous --goal joint_welfare --out test5.txt
"""

import os
import sys
import json
import random
import argparse
from openai import OpenAI

# =====================
# アイテム設定（論文Figure 3より）
# =====================

ITEM_CONFIGS = {
    3: {
        "items": {
            "I1": {"total": 3, "values": {"A1": 7, "A2": 3, "A3": 7}},
            "I2": {"total": 2, "values": {"A1": 2, "A2": 4, "A3": 6}},
            "I3": {"total": 4, "values": {"A1": 5, "A2": 7, "A3": 4}},
            "I4": {"total": 2, "values": {"A1": 6, "A2": 6, "A3": 4}},
        },
        "agents": ["A1", "A2", "A3"],
    },
    4: {
        "items": {
            "I1": {"total": 5, "values": {"A1": 3, "A2": 5, "A3": 4, "A4": 1}},
            "I2": {"total": 3, "values": {"A1": 2, "A2": 2, "A3": 6, "A4": 2}},
            "I3": {"total": 8, "values": {"A1": 3, "A2": 1, "A3": 2, "A4": 5}},
            "I4": {"total": 3, "values": {"A1": 4, "A2": 6, "A3": 1, "A4": 2}},
        },
        "agents": ["A1", "A2", "A3", "A4"],
    },
    5: {
        "items": {
            "I1": {"total": 4, "values": {"A1": 5, "A2": 4, "A3": 7, "A4": 8, "A5": 5}},
            "I2": {"total": 4, "values": {"A1": 1, "A2": 8, "A3": 2, "A4": 5, "A5": 6}},
            "I3": {"total": 3, "values": {"A1": 5, "A2": 5, "A3": 1, "A4": 5, "A5": 5}},
            "I4": {"total": 4, "values": {"A1": 8, "A2": 2, "A3": 8, "A4": 1, "A5": 3}},
        },
        "agents": ["A1", "A2", "A3", "A4", "A5"],
    },
}

GOAL_DESCRIPTIONS = {
    "none":          ("No Goal",        "交渉の目標は特にありません。他のエージェントと交渉してください。"),
    "joint_welfare": ("Joint Welfare",  "全エージェントの合計スコアを最大化することを目指してください。"),
    "envy_index":    ("Envy Index",     "全エージェント間の妬み指数（Envy Index）を最小化してください。Envy Indexとは「自分の束より他人の束の方が自分にとって価値が高い」と感じる差の合計です。値が小さいほど公平です。"),
    "personal_score":("Personal Score", "あなた自身のスコアを最大化することを目指してください。"),
}

# =====================
# スコア・Envy Index計算
# =====================

def calc_score(agent, allocation, items):
    return sum(items[item]["values"][agent] * cnt for item, cnt in allocation[agent].items())

def calc_envy_index(allocation, agents, items):
    total_envy = 0
    for i in agents:
        vi_Bi = calc_score(i, allocation, items)
        for j in agents:
            if i == j:
                continue
            vi_Bj = sum(items[item]["values"][i] * cnt for item, cnt in allocation[j].items())
            total_envy += max(0, vi_Bj - vi_Bi)
    return total_envy

def validate_allocation(allocation, agents, items):
    try:
        for item, info in items.items():
            if sum(allocation[ag][item] for ag in agents) != info["total"]:
                return False
        for ag in agents:
            for item in items:
                if not isinstance(allocation[ag][item], int) or allocation[ag][item] < 0:
                    return False
        return True
    except (KeyError, TypeError):
        return False

# =====================
# プロンプト生成
# =====================

def build_prompt(agent, round_num, past_responses, dialogue,
                 agents, items, decision_rule, goal, max_rounds, last_proposal):

    # 1. Role
    role = f"## あなたの役割\nあなたはエージェント {agent} です。\n参加エージェント: {', '.join(agents)}\n\nあなたのアイテム価値表（1個あたりの点数）:\n"
    for item, info in items.items():
        role += f"  {item}: {info['values'][agent]} 点\n"
    role += "\nアイテムの総個数:\n"
    for item, info in items.items():
        role += f"  {item}: {info['total']} 個\n"

    # 2. Goal
    goal_name, goal_desc = GOAL_DESCRIPTIONS[goal]
    goal_text = f"\n## 交渉ゴール\nゴール名: {goal_name}\n{goal_desc}\n"

    # 3. Rule
    rule_text = f"""
## ルール
- 最大ラウンド数: {max_rounds}
- 決定ルール: {"多数決（過半数が受け入れれば合意成立）" if decision_rule == "majority" else "全会一致（全員が受け入れれば合意成立）"}
- 各ラウンドで、あなたは以下を決定してください:
  1. 発言するかどうか（Willingness）
  2. 直前の提案を受け入れるかどうか（AcceptLastProposal）
  3. 新しい分配案（Allocation）
  4. 他のエージェントへのメッセージ（Statement）
- Allocationは全アイテムを全エージェントに過不足なく割り当てること。
"""

    # 4. 直前の提案と自分が受け入れた場合のスコア（差異3の修正）
    if last_proposal:
        score_if_accept = calc_score(agent, last_proposal, items)
        proposal_text = f"\n## 現在テーブルにある提案（AcceptLastProposalの対象）\n"
        for ag, alloc in last_proposal.items():
            items_str = ", ".join(f"{it}×{cnt}" for it, cnt in alloc.items() if cnt > 0)
            score = items[list(items.keys())[0]]["values"].get(ag, 0)  # dummy
            score_for_me = sum(items[it]["values"][agent] * cnt for it, cnt in alloc.items())
            proposal_text += f"  {ag} ← {items_str}  （あなた({agent})目線の価値: {score_for_me}点）\n"
        proposal_text += f"\n  ★ この提案を受け入れた場合、あなたのスコアは {score_if_accept} 点になります。\n"
    else:
        proposal_text = "\n## 現在テーブルにある提案\nまだ提案がありません。\n"

    # 5. Past Response
    past = "\n## あなたの過去の応答（他のエージェントには非公開）\n"
    if past_responses:
        for i, r in enumerate(past_responses, 1):
            past += f"ラウンド{i}: {r}\n"
    else:
        past += "（まだ履歴はありません）\n"

    # 6. Dialogue
    dialogue_text = "\n## 公開された対話履歴\n"
    if dialogue:
        for entry in dialogue:
            dialogue_text += f"- {entry}\n"
    else:
        dialogue_text += "（まだ対話はありません）\n"

    # 7. Task
    task = f"\n## タスク\n現在はラウンド {round_num} です。上記の情報をもとに行動してください。\n"

    # 8. Response Format
    example_alloc = {ag: {item: 0 for item in items} for ag in agents}
    response_format = f"""
## 返答フォーマット
以下のJSONのみを返してください。説明文や```は不要です。

{{
  "Reflection": "現在の状況の分析と戦略（自然言語）",
  "Willingness": true,
  "AcceptLastProposal": false,
  "Allocation": {json.dumps(example_alloc, ensure_ascii=False)},
  "Statement": "他のエージェントへのメッセージ（自然言語）"
}}

注意:
- Allocationは全アイテムを全エージェントに過不足なく割り当てること
- Willingness=trueのときは、AllocationかStatementの少なくとも一方を空にしないこと
- AcceptLastProposalはテーブルにある提案を受け入れるかどうかのbool値
"""
    return role + goal_text + rule_text + proposal_text + past + dialogue_text + task + response_format

# =====================
# LLM呼び出し
# =====================

def call_llm(prompt, model, client):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = response.choices[0].message.content.strip()
        # ```json ... ``` 形式でも対応
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        parsed["_tokens"] = response.usage.total_tokens
        return parsed
    except Exception as e:
        return {"_error": str(e), "Reflection": f"エラー: {e}",
                "Willingness": False, "AcceptLastProposal": False,
                "Allocation": None, "Statement": "", "_tokens": 0}

# =====================
# 出力ファイルの自動採番
# =====================

def get_output_path(out_arg):
    if out_arg:
        return out_arg
    i = 1
    while os.path.exists(f"test{i}.txt"):
        i += 1
    return f"test{i}.txt"

# =====================
# メイン交渉ループ
# =====================

def run_negotiation(args, out_path):
    n = args.agents
    config = ITEM_CONFIGS.get(n)
    if config is None:
        print(f"エラー: --agents は 3〜5 の整数を指定してください（現在: {n}）")
        sys.exit(1)

    items  = config["items"]
    agents = config["agents"]
    decision_rule  = args.rule
    goal           = args.goal
    max_rounds     = args.rounds
    model          = args.model

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    lines = []  # ファイルに書く行を蓄積

    def log(text=""):
        print(text)
        lines.append(text)

    # ─── ゲーム設定の表示 ───
    log(f"\n{'='*60}")
    log("【ゲーム設定】")
    log(f"{'='*60}")
    log(f"■ 参加エージェント: {', '.join(agents)}")
    log(f"■ 決定ルール: {'多数決（過半数が受け入れれば合意成立）' if decision_rule == 'majority' else '全会一致（全員が受け入れれば合意成立）'}")
    log(f"■ ラウンド上限: {max_rounds} ラウンド")
    log(f"■ 交渉ゴール: {GOAL_DESCRIPTIONS[goal][0]}")
    log(f"  └ {GOAL_DESCRIPTIONS[goal][1]}")
    log(f"■ モデル: {model}")
    log(f"■ 出力ファイル: {out_path}")

    log(f"\n■ アイテム設定（各エージェントにとっての1個あたりの価値）")
    header = f"  {'アイテム':>6} | {'個数':>4} | " + " | ".join(f"{a:>4}" for a in agents)
    log(header)
    log("  " + "-" * (len(header) - 2))
    for item, info in items.items():
        vals = " | ".join(f"{info['values'][a]:>4}" for a in agents)
        log(f"  {item:>6} | {info['total']:>4} | {vals}")

    log(f"\n■ 各エージェントが全アイテムを独占した場合の最大スコア（参考）")
    for ag in agents:
        mx = sum(info["total"] * info["values"][ag] for info in items.values())
        log(f"  {ag}: {mx} 点")

    log(f"\n{'='*60}")
    log("交渉開始")
    log(f"{'='*60}")

    # ─── 初期化 ───
    past_responses = {ag: [] for ag in agents}
    dialogue       = []
    last_proposal  = None
    total_tokens   = 0

    result = None

    for round_num in range(1, max_rounds + 1):
        log(f"\n{'─'*50}")
        log(f"ラウンド {round_num} / {max_rounds}")
        log(f"{'─'*50}")

        responses = {}

        for agent in agents:
            prompt = build_prompt(agent, round_num, past_responses[agent], dialogue,
                                  agents, items, decision_rule, goal, max_rounds, last_proposal)
            res = call_llm(prompt, model, client)
            if res.get("Willingness"):
                if not res.get("Allocation") and not res.get("Statement", "").strip():
                    res["Willingness"] = False
            responses[agent] = res
            total_tokens += res.get("_tokens", 0)

            log(f"\n  ┌─ {agent} の判断 {'─'*30}")
            log(f"  │ 思考: {res.get('Reflection', '')}")
            log(f"  │ 前の提案を受け入れる: {'Yes ✓' if res.get('AcceptLastProposal') else 'No ✗'}")
            log(f"  │ 発言する: {'Yes' if res.get('Willingness') else 'No（沈黙）'}")
            alloc = res.get("Allocation")
            if alloc and res.get("Willingness"):
                log(f"  │ 提案する分配案:")
                for ag, ag_alloc in alloc.items():
                    score_v = sum(items[it]["values"][agent] * cnt for it, cnt in ag_alloc.items())
                    items_str = ", ".join(f"{it}×{cnt}" for it, cnt in ag_alloc.items() if cnt > 0)
                    log(f"  │   {ag} ← {items_str or 'なし'}  （{agent}目線: {score_v}点）")
            if res.get("Statement", "").strip():
                log(f"  │ メッセージ: {res['Statement']}")
            log(f"  └{'─'*42}")

        # 合意判定
        accept_votes = {ag: responses[ag].get("AcceptLastProposal", False) for ag in agents}
        accept_count = sum(accept_votes.values())
        yes_list = [a for a, v in accept_votes.items() if v]
        no_list  = [a for a, v in accept_votes.items() if not v]

        log(f"\n  ▼ 合意判定")
        if last_proposal is None:
            log(f"    まだ提案がないため合意判定なし")
        else:
            log(f"    現在の提案: {json.dumps(last_proposal, ensure_ascii=False)}")
            log(f"    受け入れ: {yes_list or 'なし'}  |  拒否: {no_list or 'なし'}")
            needed = len(agents) // 2 + 1 if decision_rule == "majority" else len(agents)
            log(f"    票数: {accept_count}/{len(agents)}  （{'多数決: ' + str(needed) + '票以上で成立' if decision_rule == 'majority' else '全会一致: 全員で成立'}）")

            agreed = (decision_rule == "majority" and accept_count > len(agents) / 2) or \
                     (decision_rule == "unanimous" and accept_count == len(agents))
            if agreed:
                log(f"    → 【合意成立！】")
                result = _make_result(True, round_num, last_proposal, total_tokens, agents, items)
                break
            else:
                log(f"    → 合意ならず、交渉継続")

        # 発言者選出
        willing = [ag for ag in agents if responses[ag].get("Willingness", False)]
        log(f"\n  ▼ 今ラウンドの発言")
        if willing:
            speaker  = random.choice(willing)
            s_alloc  = responses[speaker].get("Allocation")
            s_stmt   = responses[speaker].get("Statement", "")
            log(f"    発言者: {speaker}  （発言希望: {willing} → ランダム選出）")

            if s_alloc and validate_allocation(s_alloc, agents, items):
                last_proposal = s_alloc
                dialogue.append(f"[R{round_num}] {speaker} が提案: {json.dumps(s_alloc, ensure_ascii=False)}")
                parts = []
                for ag, ag_alloc in s_alloc.items():
                    s = ", ".join(f"{it}×{cnt}" for it, cnt in ag_alloc.items() if cnt > 0)
                    parts.append(f"{ag}←[{s}]")
                log(f"    提案: 有効 → {' '.join(parts)}")
            else:
                dialogue.append(f"[R{round_num}] {speaker} の提案は無効でした")
                log(f"    提案: 無効（アイテム合計数の不整合）→ 破棄")

            if s_stmt:
                dialogue.append(f"[R{round_num}] {speaker} のメッセージ: {s_stmt}")
                log(f"    全体メッセージ: 「{s_stmt}」")
        else:
            log(f"    全員沈黙（発言希望者なし）")

        for ag in agents:
            r = responses[ag]
            past_responses[ag].append(
                f"R{round_num}: Accept={r.get('AcceptLastProposal')} Willing={r.get('Willingness')} | {r.get('Reflection','')[:80]}"
            )

    if result is None:
        log(f"\n  → ラウンド上限到達。合意失敗。")
        result = _make_result(False, max_rounds, None, total_tokens, agents, items)

    # ─── 最終結果 ───
    log(f"\n{'='*60}")
    log("【最終結果】")
    log(f"{'='*60}")
    log(f"  合意:         {'成功 ✓' if result['success'] else '失敗 ✗'}")
    log(f"  ラウンド数:   {result['rounds']}")
    log(f"  使用トークン: {result['tokens']}")
    if result["success"]:
        log(f"  各エージェントのスコア: {result['scores']}")
        log(f"  合計スコア:   {result['total_score']}")
        log(f"  Envy Index:   {result['envy_index']:.2f}")
        log(f"  分配案: {json.dumps(result['allocation'], ensure_ascii=False)}")

    # ─── ファイル書き出し ───
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n→ ログを {out_path} に保存しました。")

    return result


def _make_result(success, rounds, allocation, tokens, agents, items):
    result = {"success": success, "rounds": rounds, "allocation": allocation,
              "tokens": tokens, "scores": {}, "total_score": 0, "envy_index": None}
    if success and allocation:
        for ag in agents:
            result["scores"][ag] = calc_score(ag, allocation, items)
        result["total_score"] = sum(result["scores"].values())
        result["envy_index"]  = calc_envy_index(allocation, agents, items)
    return result

# =====================
# エントリポイント
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMエージェント交渉シミュレーション")
    parser.add_argument("--rule",   default="majority",      choices=["majority","unanimous"])
    parser.add_argument("--goal",   default="joint_welfare", choices=["none","joint_welfare","envy_index","personal_score"])
    parser.add_argument("--rounds", default=20,  type=int)
    parser.add_argument("--agents", default=3,   type=int)
    parser.add_argument("--model",  default="gpt-4o-mini")
    parser.add_argument("--out",    default=None, help="出力ファイル名（省略時は自動採番）")
    args = parser.parse_args()

    out_path = get_output_path(args.out)
    run_negotiation(args, out_path)
