# TimeMMD Aurora-Standard Benchmark

This harness evaluates Chronos-2 and Chronos-2-ECHO using Aurora's TimeMMD protocol:
9 domains, `features=S`, `target=OT`, 7/1/2 split, and scaled-space metrics.

## Setup

```sh
D:\envs\chronos\python.exe -m pip install -e ".[timemmd]"
```

## Get Data

Use the Aurora benchmark data, not a locally re-merged Time-MMD export. Aurora's
official repo links the benchmark datasets here:
https://drive.google.com/file/d/12tJk858WaoG7ZVSvUq8KU1oHfGNJrARF/view?usp=drive_link

Download and extract it, then point `--data-root` at the extracted TimeMMD
dataset directory containing the nine CSV files below.

## Data Layout

Pass a directory containing Aurora-compatible CSVs:

```text
Agriculture.csv
Climate.csv
Economy.csv
Energy.csv
Environment.csv
Health.csv
Security.csv
Traffic.csv
SocialGood.csv
```

Each CSV must contain `date`, `OT`, `prior_history_avg`, `start_date`, `end_date`, and `fact`.
Missing `fact` values are handled like Aurora's loader: `NaN` becomes `No information available`.
The default ECHO text tokenizer is bundled at `TimeMMD\aurora\bert_config`.
Pass `--text-tokenizer` explicitly when using a different tokenizer path.

## Dry Run

```sh
D:\envs\chronos\python.exe -m TimeMMD.run_benchmark --data-root D:\path\to\TimeMMD\dataset --dry-run
```

## Full Run

```sh
D:\envs\chronos\python.exe -m TimeMMD.run_benchmark --data-root D:\path\to\TimeMMD\dataset
```

The default run matches Aurora's zero-shot script and evaluates `chronos2,echo_zero_shot`.
Outputs are written to `TimeMMD\runs\<run_id>\`.

Few-shot ECHO is explicit:

```sh
D:\envs\chronos\python.exe -m TimeMMD.run_benchmark --data-root D:\path\to\TimeMMD\dataset --models echo_few_shot
```

The released Aurora TimeMMD data has no 10% few-shot training windows for Energy/Security under
the Aurora seq_len matrix, so `echo_few_shot` is rejected unless the manifest excludes those tasks.
Few-shot checkpoints are cached under `TimeMMD\checkpoints\chronos2_echo_fewshot\`.

Aurora reference values are the Aurora columns from Table 12 of
https://arxiv.org/html/2509.22295v4.
