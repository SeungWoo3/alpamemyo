"""
7GB VRAM 제한 조건 재측정 스크립트

기존 결과에서 7GB가 4.76s로 baseline(4.79s)보다 빨라 이상치로 판단.
재측정하여 올바른 값을 확인한다.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_vram_limit_test import run_single_experiment

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("=" * 70)
    print("7GB VRAM Limit Re-measurement")
    print("=" * 70)
    print("Previous 7GB result: inference_time=4.76s (suspicious)")
    print()

    # Run 7GB test
    result = run_single_experiment(7)

    # Print result
    print(f"\n{'=' * 70}")
    print("RE-MEASUREMENT RESULT")
    print(f"{'=' * 70}")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Save individual result
    retest_path = os.path.join(RESULTS_DIR, "retest_7gb_result.json")
    with open(retest_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {retest_path}")

    # Update main results.json
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "r") as f:
        all_results = json.load(f)

    # Find and replace the 7GB entry
    replaced = False
    for i, r in enumerate(all_results):
        if r["vram_limit_gb"] == 7:
            old_infer = r.get("inference_time")
            all_results[i] = result
            replaced = True
            print(f"\nUpdated results.json: 7GB entry replaced")
            print(f"  Old inference_time: {old_infer}s")
            print(f"  New inference_time: {result.get('inference_time')}s")
            break

    if not replaced:
        print("\n[WARNING] No 7GB entry found in results.json, appending.")
        all_results.append(result)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
