"""
bargaining_sim/main.py
実行エントリポイント

使い方:
  python main.py                  # LLMなし（human × bayesian のみ）デフォルト実行
  python main.py --with-llm       # LLMエージェントも含めて実行（APIキー必要）
  python main.py --colors 3       # 3色バリアント
  python main.py --games 10       # 各構成10ゲーム
  python main.py --verbose        # 1ゲームの詳細表示
"""
import argparse
import os
from game import COLORS_2, COLORS_3, COLORS_4
from simulator import run_game, run_experiments


def main():
    parser = argparse.ArgumentParser(description="バーゲニングゲーム シミュレーション")
    parser.add_argument("--with-llm",  action="store_true",
                        help="LLMエージェントを含める（OPENAI_API_KEY が必要）")
    parser.add_argument("--colors",    type=int, default=2, choices=[2, 3, 4],
                        help="チップの色数 (2/3/4)")
    parser.add_argument("--games",     type=int, default=3,
                        help="各エージェント構成で実行するゲーム数")
    parser.add_argument("--verbose",   action="store_true",
                        help="各ターンの詳細を表示")
    parser.add_argument("--demo",      action="store_true",
                        help="1ゲームだけ詳細表示（動作確認用）")
    args = parser.parse_args()

    # 色リストの選択
    colors = {2: COLORS_2, 3: COLORS_3, 4: COLORS_4}[args.colors]

    print("=" * 60)
    print("  バーゲニングゲーム シミュレーション")
    print(f"  色バリアント: {colors}  ゲーム数/構成: {args.games}")
    print("=" * 60)

    # ─── デモモード ────────────────────────────────────
    if args.demo:
        if args.with_llm:
            print("\n【デモ: LLM × Human × Bayesian (1ゲーム詳細)】")
            run_game(["llm", "human", "bayesian"], colors=colors, seed=42, verbose=True)
        else:
            print("\n【デモ: Human × Human × Bayesian (1ゲーム詳細)】")
            run_game(["human", "human", "bayesian"], colors=colors, seed=42, verbose=True)
        return

    # ─── 実験構成の定義 ───────────────────────────────
    experiments = [
        {
            "label": "Human × Human × Human",
            "types": ["human", "human", "human"],
        },
        {
            "label": "Bayesian × Bayesian × Bayesian",
            "types": ["bayesian", "bayesian", "bayesian"],
        },
        {
            "label": "Human × Human × Bayesian",
            "types": ["human", "human", "bayesian"],
        },
    ]

    if args.with_llm:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("⚠ OPENAI_API_KEY が設定されていません。LLM実験をスキップします。")
        else:
            experiments += [
                {
                    "label": "LLM × LLM × LLM",
                    "types": ["llm", "llm", "llm"],
                },
                {
                    "label": "LLM × LLM × Bayesian",
                    "types": ["llm", "llm", "bayesian"],
                },
                {
                    "label": "Human × LLM × Bayesian",
                    "types": ["human", "llm", "bayesian"],
                },
            ]

    # ─── 実験実行 ─────────────────────────────────────
    results = []
    for exp in experiments:
        print(f"\n\n{'#'*60}")
        print(f"# {exp['label']}")
        print(f"{'#'*60}")
        summary = run_experiments(
            exp["types"],
            colors=colors,
            n_games=args.games,
            verbose_each=args.verbose,
        )
        results.append((exp["label"], summary))

    # ─── 最終比較表 ───────────────────────────────────
    print(f"\n\n{'='*60}")
    print("【結果比較表】")
    print(f"{'構成':<35} {'平均サープラス':>14} {'成立率':>8}")
    print("-" * 60)
    for label, s in results:
        print(f"{label:<35} {s['mean_total_surplus']:>10.3f}±{s['std_total_surplus']:.2f}  "
              f"{s['mean_accepted_rate']:>7.1%}")


if __name__ == "__main__":
    main()
