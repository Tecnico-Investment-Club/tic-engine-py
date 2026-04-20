# Live Strategy — Input / Output & Execution

## What the strategy receives

The live script (`live_final.py`) requires:

1) A continuously updated CSV file containing:
- Open time (timestamp)
- Open
- High
- Low
- Close
- Volume

The strategy assumes BTC hourly bars that are in chronological order.


2) Two trained model files located in:
models/best_primary_model_all_features.pkl
models/best_meta_model_all_features.pkl

The primary model outputs trade direction.
The meta model outputs position size and probability.

---

## What the strategy outputs

Every cycle (runs every 5 seconds), the script computes a trading signal:

→ bet_size

This is the main output of the strategy and corresponds to the long position size that the strategy should hold at that moment.

For example, the bet size should be interpreted in the following way:
- 0.0 → flat (no position)
- 0.5 → invest 50% of the capital
- 1.0 → invest 100% of the capital

Changes on the bet size value correspond to the percentage of the capital that should be bought/sold when the position is adjusted.


Additionally:
- Detected CUSUM events are appended to `cusum_events.csv`


## How to run the live script

From the root folder:

python live_final.py

Make sure that all the files necessary for the dependencies are in the same directory.




