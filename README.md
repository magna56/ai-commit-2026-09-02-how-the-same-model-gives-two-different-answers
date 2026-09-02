# How the Same Model Gives Two Different Answers

**TL;DR:** Two machines running identical weights at temperature zero can return different answers, and neither is broken. Addition on a computer depends on the order you do it in, and the order is not recorded in the checkpoint.

Published from [The AI Commit](https://theaicommit.com/#2026-09-02/code) — AI in Production, 2026-09-02.

## Run

```bash
python3 code_example.py
```

## Output

```
One dot product, five reduction orders:

  sequential          255.57010934394964
  split-K=2           255.57010934394964   delta +0.000e+00
  split-K=4           255.57010934394958   delta -5.684e-14
  split-K=8           255.57010934394958   delta -5.684e-14
  split-K=16          255.57010934394964   delta +0.000e+00
  pairwise (tree)     255.57010934394958   delta -5.684e-14

  Same 64 numbers every time. Only the order changed.

Two candidate tokens scored from the same numbers, differently arranged.
Their exact sums are equal; only rounding can separate them.

  reduction order       token A                token B              A-B        argmax
  sequential             -290.18289845865337    -290.18289845865326   -1.14e-13   B
  pairwise (tree)        -290.18289845865337    -290.18289845865337   +0.00e+00   tie
  split-K=4              -290.18289845865337    -290.18289845865337   +0.00e+00   tie
  split-K=8              -290.18289845865337     -290.1828984586534   +5.68e-14   A

  Same numbers, same model, temperature zero. The winner changes with
  the reduction order alone.

Temperature zero was honoured in both cases. Greedy decoding is
deterministic GIVEN a reduction order, and the order is set by your
kernel, your batch shape and your parallelism layout — none of which
live in the checkpoint. That is the whole bug, and everything after
this token diverges.

```

## Code

See [`code_example.py`](code_example.py).
