"""Download the public data the experiments need, into data/ (gitignored).

uv run python experiments/setup_data.py            # everything (about 6 GB, mostly the BrowseComp-Plus corpus)
uv run python experiments/setup_data.py oolong     # or one step: oolong, pairs, infinitebench, trec, books, bcp

oolong         OOLONG-synth validation split + the test split at 32k/131k/262k   (E1-E4, E6, E9-E11)
pairs          OOLONG-Pairs questions at 32k and 131k                             (E2, E3, E8b)
infinitebench  InfiniteBench En.MC                                                (E7)
trec           the original TREC question-classification files                    (E8f, E8g)
books          14 Project Gutenberg novels, checked against SHA-256               (E5)
bcp            BrowseComp-Plus document corpus (the questions load at run time)   (E12)
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

DATA = Path("data")

BOOKS = {  # Gutenberg id -> sha256 of the file used in the write-up
    "108": "8c4de14b3a1952417eca4673fcb1b8aa8f946f8a4164f10ec761368af34d5e5a",
    "1260": "13414dee2951c3ee731d76d2ffd822016b2479c892162760c5d0eb2aa5fa7631",
    "1342": "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75",
    "1400": "9a637118af8e953e9764ec603d9b0a032883384d465acac2e27966a80cf1c6f8",
    "145": "dc2e0107e6ae07e8e33da934b2ec4e8e600826a2e0cd48df265119cd02e5fb50",
    "158": "532b122b4e6a76cc556d6fdcd729b5892f5c4ce4a1b7060b9f832adfad8680dc",
    "161": "22272ec4d4da2f50cda51edf34ab8486b325c4a99580120db565fb8917228a22",
    "1661": "922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0",
    "2097": "4cdea89cf6cd2567a556d0e6901edb89949dd79e200dbaf4ced4cabf1d5d2c26",
    "244": "8bbec71f9fa9c2acc9afe9be70560ca5f65c1bc95ef5af8252e9097f8d012497",
    "2852": "f7c8c68729d32a5f6a04a5a9ad73d184176e66a6c76b3992c41c1ab47b427d98",
    "766": "51650d348820a64ca4d4233da035f804819d2231801b59373db4e1279a1985e0",
    "768": "e533fe750589f0421d5d744576315f5c2b9b0d69e981179ea0551bbf134c5e02",
    "98": "d54c2b80d40a40b982cd88852c6180bb944d95acdb028af3d0e01a1750681784",
}


def books(dest=DATA / "books"):
    dest.mkdir(parents=True, exist_ok=True)
    bad = []
    for gid, sha in BOOKS.items():
        f = dest / f"{gid}.txt"
        if not f.exists():
            url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
            urllib.request.urlretrieve(url, f)
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        status = "ok" if got == sha else "CHANGED"
        if got != sha:
            bad.append(gid)
        print(f"{gid:>5} {status}")
    if bad:
        print(
            f"WARNING: {len(bad)} book(s) differ from the versions used in the write-up: {bad}. "
            "Results on those files may not match exactly."
        )


def _get(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    print("ok", dest)


def oolong():
    from datasets import load_dataset

    if not (DATA / "oolong_synth_val.parquet").exists():
        load_dataset("oolongbench/oolong-synth", split="validation").to_parquet(DATA / "oolong_synth_val.parquet")
    if not (DATA / "oolong_synth_test_32k_262k.parquet").exists():
        test = load_dataset("oolongbench/oolong-synth", split="test")
        test.filter(lambda r: r["context_len"] in (32768, 131072, 262144)).to_parquet(
            DATA / "oolong_synth_test_32k_262k.parquet"
        )
    print("ok oolong")


def pairs():
    for L in (32768, 131072):
        _get(
            f"https://huggingface.co/datasets/mit-oasys/oolong-pairs/resolve/main/data/oolong-pairs-{L}.json",
            DATA / f"oolong-pairs-{L}.json",
        )


def infinitebench():
    _get(
        "https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/main/longbook_choice_eng.jsonl",
        DATA / "longbook_choice_eng.jsonl",
    )


def trec():
    for f in ("train_5500.label", "TREC_10.label"):
        _get(f"https://cogcomp.seas.upenn.edu/Data/QA/QC/{f}", DATA / "trec" / f)


def bcp():
    from datasets import load_dataset

    out = DATA / "bcp" / "corpus.parquet"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        load_dataset("Tevatron/browsecomp-plus-corpus", split="train").to_parquet(out)
    print("ok", out)


STEPS = {"oolong": oolong, "pairs": pairs, "infinitebench": infinitebench, "trec": trec, "books": books, "bcp": bcp}

if __name__ == "__main__":
    unknown = [n for n in sys.argv[1:] if n not in STEPS]
    if unknown:
        sys.exit(f"unknown step(s) {unknown}; choose from {list(STEPS)}")
    for name in sys.argv[1:] or STEPS:
        STEPS[name]()
