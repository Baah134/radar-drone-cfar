"""Print and save derived radar quantities. Usage: python -m scripts.radar_params"""
import json
from pathlib import Path

from src.radar.params import FMCW, ClassDwell


def main():
    d = {"detection_waveform": FMCW().derived(), "classification_dwell": ClassDwell().derived()}
    Path("results").mkdir(exist_ok=True)
    Path("results/radar_params.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    for k, v in d.items():
        print(k)
        for a, b in v.items():
            print(f"  {a:34s} {b:.6g}" if isinstance(b, float) else f"  {a:34s} {b}")


if __name__ == "__main__":
    main()
