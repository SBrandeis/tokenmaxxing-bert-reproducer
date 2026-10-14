# Tokenmaxxing BERT: reproducer

Measurements behind the post: 🤗 tokenizers `main` (`fb49a2922`) vs `perf/faster-unicode-normalizers` (`5b5a3fcc1`), plus 0.23.2.

| folder | what |
|---|---|
| `benchmark/` | encode throughput, normalizer cost by input size, profiles, releases (`start_jobs.sh`) |
| `census/` | normalizers and tokenizer types of the top 10,000 Hub models |
| `ascii-share/` | share of ASCII in each benchmark corpus |

Results: `benchmark/results/` (tables in `summary.md`), `ascii-share/ascii_share.json`. `make census-fetch` writes the census to `census/results/`.

## Local (Docker)

```sh
make local                              # all parts → local/<stamp>/
make local PARTS=machine,versions SECS=2
```

## HF Jobs

> [!WARNING]
> These targets launch [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) on your account: paid compute, billed per minute.
> `make jobs` runs for about 45 min on `cpu-performance` ($1.90/h, ≈ $1.50).
> `make census` takes about 6 min on `cpu-upgrade` (< $0.01).

```sh
make jobs                    # full run + releases comparison, on cpu-performance
make job PARTS=machine,throughput
make fetch ID=<job-id>       # → benchmark/results/<job-id>/
make census                  # then: make census-fetch ID=<job-id>
```

`start_jobs.sh` settings: `PARTS` (`machine,throughput,sizes,profile,versions`), `MAIN`, `BRANCH`, `ROUNDS`, `SECS`, `SIZE_ROUNDS`.
