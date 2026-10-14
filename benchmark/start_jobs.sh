#!/usr/bin/env bash
# Tokenmaxxing BERT benchmark: tokenizers main vs perf/faster-unicode-normalizers (and 0.23.2).
# Self-contained; runs in rust:1-bookworm with hf-internal-testing/tokenizers-test-data at /data.
set -euo pipefail
MAIN=${MAIN:-fb49a29223724a96f26e061bec8ec7517aa14177}
BRANCH=${BRANCH:-5b5a3fcc18335eed616ae89adb714a048fed1a31}
ROUNDS=${ROUNDS:-5}          # timed rounds, after one warm-up
SECS=${SECS:-5}              # seconds per throughput run
SIZE_ROUNDS=${SIZE_ROUNDS:-4}
PARTS=${PARTS:-machine,throughput,sizes,profile}   # also: versions (0.23.2 / main / branch)
D=/data/fixtures
has() { [[ ",$PARTS," == *",$1,"* ]]; }

echo "== start $(date -u +%FT%TZ) parts=$PARTS rounds=$ROUNDS secs=$SECS flavor=${FLAVOR:-?}"
CPU0=$(grep Cpus_allowed_list /proc/self/status | cut -f2 | cut -d, -f1 | cut -d- -f1)
PIN=""; command -v taskset >/dev/null && taskset -c "$CPU0" true 2>/dev/null && PIN="taskset -c $CPU0"
steal() { awk '/^cpu /{printf "%.0f %.0f\n", $9, $2+$3+$4+$5+$6+$7+$8+$9}' /proc/stat; }

if has machine; then
  echo "== machine.json begin"
  {
    echo '{'
    echo "\"flavor\": \"${FLAVOR:-unknown}\","
    echo "\"date\": \"$(date -u +%FT%TZ)\","
    echo "\"kernel\": \"$(uname -r)\","
    echo "\"nproc\": $(nproc),"
    echo "\"cpus_allowed\": \"$(grep Cpus_allowed_list /proc/self/status | cut -f2)\","
    echo "\"cgroup_cpu_max\": \"$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)\","
    echo "\"memory_kb\": $(awk '/MemTotal/{print $2}' /proc/meminfo),"
    echo "\"simd_flags\": \"$(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(sse4_2|avx|avx2|avx512f|avx512bw|avx512vl|avx512_vbmi2|bmi2|popcnt)$' | tr '\n' ' ')\","
    echo "\"pinned_to\": \"${PIN:-none}\","
    echo "\"lscpu\": $(lscpu -J)"
    echo '}'
  }
  echo "== machine.json end"
fi

# Build main and branch: encode, sizes and profile binaries.
git clone -q --filter=blob:none https://github.com/huggingface/tokenizers /src
for arm in main branch; do
  sha=$MAIN; [ $arm = branch ] && sha=$BRANCH
  git -C /src worktree add -q /w/$arm $sha
  echo "== commit $arm $(git -C /w/$arm rev-parse HEAD)"
  mkdir -p /h/$arm/src/bin
  cat > /h/$arm/Cargo.toml <<TOML
[package]
name = "nbench"
version = "0.0.0"
edition = "2024"
[dependencies]
tokenizers = { path = "/w/$arm/tokenizers", default-features = false, features = ["regex"] }
pprof = { version = "=0.14.1", default-features = false }
[profile.release]
lto = "fat"
codegen-units = 1
debug = true  # names inlined frames in the profiles
[workspace]
TOML
  cp /w/$arm/tokenizers/Cargo.lock /h/$arm/
  cat > /h/$arm/src/bin/encode.rs <<'RS'
//! End-to-end encode throughput, one thread: `encode <tokenizer.json> <corpus.txt> <seconds>`.
//! Prints MB/s plus an FNV-1a hash of the token ids, so two builds can be checked for same tokens out.
//! `encode nfd <in> <out>` writes the NFD form of a file (the "decomposed" corpus).
use std::time::{Duration, Instant};
use tokenizers::pipeline::{EncodeOptions, Normalizer};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a[0] == "nfd" {
        let text = std::fs::read_to_string(&a[1]).unwrap();
        let out = tokenizers::normalizers::NFD.normalize(&text, 0).unwrap();
        std::fs::write(&a[2], out.as_bytes()).unwrap();
        return;
    }
    let json = tokenizers::canonicalize_file(&a[0]).unwrap();
    let tokenizer = tokenizers::from_json(&json).unwrap();
    let text = std::fs::read_to_string(&a[1]).unwrap();
    let secs: f64 = a[2].parse().unwrap();
    tokenizers::parallelism::set_num_threads(1);
    let options = EncodeOptions::no_specials();
    let first = tokenizer.encode(text.as_str(), &options).wait().unwrap();
    let mut hash: u64 = 0xcbf29ce484222325;
    for t in first[0].ids() {
        hash = (hash ^ t.id() as u64).wrapping_mul(0x100000001b3);
    }
    let (start, mut passes) = (Instant::now(), 0usize);
    while start.elapsed() < Duration::from_secs_f64(secs) {
        std::hint::black_box(tokenizer.encode(text.as_str(), &options).wait().unwrap());
        passes += 1;
    }
    let s = start.elapsed().as_secs_f64().max(1e-9);
    println!("{:.2}\t{:016x}\t{}\t{}", (passes * text.len()) as f64 / s / 1e6, hash, first[0].ids().len(), passes);
}
RS
  cat > /h/$arm/src/bin/sizes.rs <<'RS'
//! Normalizer cost vs input size.
//! `sizes <fixtures-dir> <reps>`: for every normalizer × corpus × size, normalizes ~4 MB of distinct
//! slices `reps` times and prints the best time, plus an output hash and how many calls borrowed.
use std::hint::black_box;
use std::time::Instant;

use tokenizers::normalizers::replace::ReplacePattern;
use tokenizers::normalizers::{BertNormalizer, Lowercase, NFC, NFD, Replace, StripAccents};
use tokenizers::pipeline::Normalizer;

const TOTAL: usize = 4 << 20;
const SIZES: [usize; 10] = [8, 16, 32, 64, 128, 256, 1024, 4096, 16384, 65536];
const CORPORA: [(&str, &str); 4] = [
    ("eng_Latn", "lang/eng_Latn.txt"),
    ("rus_Cyrl", "lang/rus_Cyrl.txt"),
    ("cmn_Hani", "lang/cmn_Hani.txt"),
    ("code_mixed", "modalities/code_mixed.txt"),
];

fn slices(text: &str, size: usize) -> Vec<String> {
    let (mut out, mut total, mut start) = (Vec::new(), 0, 0);
    while total < TOTAL {
        if start + size > text.len() {
            start = 0;
        }
        let mut end = start + size;
        while !text.is_char_boundary(end) {
            end += 1;
        }
        total += end - start;
        out.push(text[start..end].to_string());
        start = end;
        while start < text.len() && !text.is_char_boundary(start) {
            start += 1;
        }
    }
    out
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    *h = (*h ^ 0xff).wrapping_mul(0x100000001b3);
}

fn main() {
    let fixtures = std::env::args().nth(1).unwrap();
    let reps: usize = std::env::args().nth(2).map_or(3, |s| s.parse().unwrap());
    let normalizers: Vec<(&str, Box<dyn Normalizer>)> = vec![
        ("bert_uncased", Box::new(BertNormalizer::new(true, true, None, true))),
        ("bert_cased", Box::new(BertNormalizer::new(true, true, None, false))),
        ("nfc", Box::new(NFC)),
        ("nfd", Box::new(NFD)),
        ("lowercase", Box::new(Lowercase)),
        ("strip_accents", Box::new(StripAccents)),
        ("replace_lit", Box::new(Replace::new(ReplacePattern::String("``".into()), "\"").unwrap())),
        ("replace_regex", Box::new(Replace::new(ReplacePattern::Regex(" {2,}".into()), " ").unwrap())),
    ];
    for (corpus, path) in CORPORA {
        let text = std::fs::read_to_string(format!("{fixtures}/{path}")).unwrap();
        for size in SIZES {
            let inputs = slices(&text, size);
            let bytes: usize = inputs.iter().map(String::len).sum();
            for (name, n) in &normalizers {
                let (mut hash, mut borrowed) = (0xcbf29ce484222325u64, 0usize);
                for input in &inputs {
                    let out = n.normalize(input, 0).unwrap();
                    borrowed += matches!(out, std::borrow::Cow::Borrowed(_)) as usize;
                    fnv(&mut hash, out.as_bytes());
                }
                let mut best = u128::MAX;
                for _ in 0..reps {
                    let start = Instant::now();
                    let mut sink = 0usize;
                    for input in &inputs {
                        sink += n.normalize(black_box(input), 0).unwrap().len();
                    }
                    black_box(sink);
                    best = best.min(start.elapsed().as_nanos());
                }
                println!("{name}\t{corpus}\t{size}\t{}\t{bytes}\t{best}\t{hash:016x}\t{borrowed}", inputs.len());
            }
        }
    }
}
RS
  cat > /h/$arm/src/bin/profile.rs <<'RS'
//! Sampled stacks of the encode loop (pprof-rs, 999 Hz): `profile <tokenizer.json> <corpus.txt> <seconds>`.
//! Prints folded stacks for flame graphs.
use std::time::{Duration, Instant};
use tokenizers::pipeline::EncodeOptions;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let json = tokenizers::canonicalize_file(&a[0]).unwrap();
    let tokenizer = tokenizers::from_json(&json).unwrap();
    let text = std::fs::read_to_string(&a[1]).unwrap();
    let secs: f64 = a[2].parse().unwrap();
    tokenizers::parallelism::set_num_threads(1);
    let options = EncodeOptions::no_specials();
    std::hint::black_box(tokenizer.encode(text.as_str(), &options).wait().unwrap());

    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(999)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .unwrap();
    let (start, mut passes) = (Instant::now(), 0usize);
    while start.elapsed() < Duration::from_secs_f64(secs) {
        std::hint::black_box(tokenizer.encode(text.as_str(), &options).wait().unwrap());
        passes += 1;
    }
    let report = guard.report().build().unwrap();
    for (frames, count) in &report.data {
        // frames.frames is leaf first; each entry lists the inlined calls of one return address.
        let stack: Vec<String> = frames.frames.iter().rev().flat_map(|f| f.iter().rev().map(|s| s.name())).collect();
        println!("{} {}", stack.join(";"), count);
    }
    let s = start.elapsed().as_secs_f64();
    println!("# {:.2} MB/s {passes} passes {s:.1} s", (passes * text.len()) as f64 / s / 1e6);
}
RS
  (cd /h/$arm && cargo build -q --release)
done
# 0.23.2 from crates.io, for the versions part (its API differs, hence its own harness).
if has versions; then
  mkdir -p /h/v023/src
  cat > /h/v023/Cargo.toml <<'TOML'
[package]
name = "enc023"
version = "0.0.0"
edition = "2021"
[dependencies]
tokenizers = { version = "=0.23.2", default-features = false, features = ["onig"] }
[profile.release]
lto = "fat"
codegen-units = 1
[workspace]
TOML
  cat > /h/v023/src/main.rs <<'RS'
//! Same as encode.rs, for the 0.23 API: `enc023 <tokenizer.json> <corpus.txt> <seconds>`.
use std::time::{Duration, Instant};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let tokenizer = tokenizers::Tokenizer::from_file(&a[0]).unwrap();
    let text = std::fs::read_to_string(&a[1]).unwrap();
    let secs: f64 = a[2].parse().unwrap();
    let first = tokenizer.encode(text.as_str(), false).unwrap();
    let mut hash: u64 = 0xcbf29ce484222325;
    for &id in first.get_ids() {
        hash = (hash ^ id as u64).wrapping_mul(0x100000001b3);
    }
    let (start, mut passes) = (Instant::now(), 0usize);
    while start.elapsed() < Duration::from_secs_f64(secs) {
        std::hint::black_box(tokenizer.encode(text.as_str(), false).unwrap());
        passes += 1;
    }
    let s = start.elapsed().as_secs_f64().max(1e-9);
    println!("{:.2}\t{:016x}\t{}\t{}", (passes * text.len()) as f64 / s / 1e6, hash, first.get_ids().len(), passes);
}
RS
  (cd /h/v023 && cargo build -q --release)
  echo "== releases built: 0.23.2 (1.0 = main $MAIN)"
fi
echo "== toolchain $(cd /h/main && rustc -V)"
enc() { $PIN /h/$1/target/release/encode "$2" "$3" "$4"; }
/h/main/target/release/encode nfd $D/lang/ell_Grek.txt /tmp/ell_Grek_nfd.txt

if has throughput; then
  read s0 t0 < <(steal)
  echo "== throughput: model corpus arm round MB/s ids_hash ntok passes"
  for model in bert-base-uncased:/data/bert-base-uncased.json modernbert-base:$D/models/modernbert-base.json; do
    m=${model%%:*}; mf=${model#*:}
    for corpus in eng_Latn:$D/lang/eng_Latn.txt code_mixed:$D/modalities/code_mixed.txt rus_Cyrl:$D/lang/rus_Cyrl.txt \
                  ell_Grek:$D/lang/ell_Grek.txt arb_Arab:$D/lang/arb_Arab.txt cmn_Hani:$D/lang/cmn_Hani.txt ell_Grek_nfd:/tmp/ell_Grek_nfd.txt; do
      c=${corpus%%:*}; cf=${corpus#*:}
      for r in $(seq 0 $ROUNDS); do  # round 0 is a warm-up, kept but tagged
        if (( r % 2 )); then order="main branch"; else order="branch main"; fi
        for arm in $order; do printf "tp\t%s\t%s\t%s\t%s\t%s\n" $m $c $arm $r "$(enc $arm $mf $cf $SECS)"; done
      done
    done
  done
  read s1 t1 < <(steal)
  awk -v a=$s0 -v b=$s1 -v c=$t0 -v d=$t1 'BEGIN{printf "== steal during throughput: %.4f%%\n", (b-a)*100/(d-c+1)}'
fi

if has versions; then
  verbin() { case $1 in v0.23.2) echo /h/v023/target/release/enc023;; main) echo /h/main/target/release/encode;; branch) echo /h/branch/target/release/encode;; esac; }
  read s0 t0 < <(steal)
  echo "== versions: model corpus arm round MB/s ids_hash ntok passes"
  for model in bert-base-uncased:/data/bert-base-uncased.json modernbert-base:$D/models/modernbert-base.json; do
    m=${model%%:*}; mf=${model#*:}
    for corpus in eng_Latn:$D/lang/eng_Latn.txt code_mixed:$D/modalities/code_mixed.txt rus_Cyrl:$D/lang/rus_Cyrl.txt cmn_Hani:$D/lang/cmn_Hani.txt; do
      c=${corpus%%:*}; cf=${corpus#*:}
      for r in $(seq 0 $ROUNDS); do
        case $((r % 3)) in 0) order="v0.23.2 main branch";; 1) order="main branch v0.23.2";; 2) order="branch v0.23.2 main";; esac
        for arm in $order; do
          printf "ver\t%s\t%s\t%s\t%s\t%s\n" $m $c $arm $r "$($PIN $(verbin $arm) $mf $cf $SECS)"
        done
      done
    done
  done
  read s1 t1 < <(steal)
  awk -v a=$s0 -v b=$s1 -v c=$t0 -v d=$t1 'BEGIN{printf "== steal during versions: %.4f%%\n", (b-a)*100/(d-c+1)}'
fi

if has sizes; then
  read s0 t0 < <(steal)
  echo "== sizes: arm round normalizer corpus size calls bytes best_ns out_hash borrowed"
  for r in $(seq 1 $SIZE_ROUNDS); do
    case $((r % 3)) in 0) order="main branch branch2";; 1) order="branch branch2 main";; 2) order="branch2 main branch";; esac
    for arm in $order; do
      bin=$arm; [ $arm = branch2 ] && bin=branch   # branch2: A/A noise control
      $PIN /h/$bin/target/release/sizes $D 3 | sed "s/^/sz\t$arm\t$r\t/"
    done
  done
  read s1 t1 < <(steal)
  awk -v a=$s0 -v b=$s1 -v c=$t0 -v d=$t1 'BEGIN{printf "== steal during sizes: %.4f%%\n", (b-a)*100/(d-c+1)}'
fi

if has profile; then
  echo "== profile"
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq valgrind >/dev/null 2>&1 || true
  echo "== stacks: model corpus arm gzip+base64(folded)"
  for c in eng_Latn rus_Cyrl; do
    for arm in main branch; do
      { $PIN /h/$arm/target/release/profile /data/bert-base-uncased.json $D/lang/$c.txt 10 | gzip -9 | base64 -w0 | sed "s/^/stacks\tbert-base-uncased\t$c\t$arm\t/"; } || true; echo
    done
  done
  # Instruction counts; irbase (1-byte input) is the tokenizer load, subtracted from ir.
  if command -v valgrind >/dev/null; then
    printf 'a' > /tmp/one.txt
    cg() {  # arm tokenizer input -> "total normalize_all_inclusive"
      valgrind --tool=callgrind --callgrind-out-file=/tmp/cg.out -q /h/$1/target/release/encode $2 $3 0 >/dev/null 2>&1 || true
      local total norm
      total=$(callgrind_annotate /tmp/cg.out 2>/dev/null | awk '/PROGRAM TOTALS/{gsub(",","",$1); print $1; exit}' || true)
      norm=$(callgrind_annotate --inclusive=yes /tmp/cg.out 2>/dev/null | grep -m1 'normalize_all' | awk '{gsub(",","",$1); print $1}' || true)
      echo "$total ${norm:-0}"
    }
    for model in bert-base-uncased:/data/bert-base-uncased.json modernbert-base:$D/models/modernbert-base.json; do
      m=${model%%:*}; mf=${model#*:}
      for arm in main branch; do printf "irbase\t%s\t%s\t%s\n" $m $arm "$(cg $arm $mf /tmp/one.txt | tr ' ' '\t')"; done
      for c in eng_Latn rus_Cyrl; do
        { head -c 1000000 $D/lang/$c.txt | iconv -c -f utf-8 -t utf-8 > /tmp/$c.1m.txt; } || true  # -c exits 1 on the cut char
        for arm in main branch; do printf "ir\t%s\t%s\t%s\t%s\n" $m $c $arm "$(cg $arm $mf /tmp/$c.1m.txt | tr ' ' '\t')"; done
      done
    done
  fi
fi
echo "== done $(date -u +%FT%TZ)"
