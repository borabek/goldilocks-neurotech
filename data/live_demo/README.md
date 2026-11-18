# One night, for the live demo

`sub-4.npz` is a single BOAS night as the live path consumes it: 30-second epochs, two headband
channels, decimated to 64 Hz, in microvolts, with the frozen reference labels and quality mask
alongside. It is here so the streaming demo works on any machine with nothing else to copy:

```bash
python live_inference.py --replay data/live_demo/sub-4.npz --cache data/live_demo
```

It prints a warm-up, then one line per 30-second epoch with the predicted stage, the model's
confidence and the running sleep total. Add `--realtime` to make it run at wall-clock speed, which
is what a demo audience should see.

BOAS is CC0 via OpenNeuro; see `CREDITS.md`. This is a derived, decimated copy of one recording,
not the raw dataset. The full epoch caches used for training are not shipped: they are 723 MB and
are only needed to retrain, not to run or to reproduce the frozen evaluation.
