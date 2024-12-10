About
=====

Use this repo to iterate on models predicting funding for early-stage startups.

Quickstart
==========

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)

Install dependencies:
```
git clone https://github.com/shahin/joinprospect-funding-prediction.git
cd joinprospect-funding-prediction
uv sync
```

Save the data from the Google Sheet as a TSV (File > Download > Tab Separated Values) and copy it to the cloned folder:
```
cp ~/Downloads/Early\ stage\ fundraising\ data\ -\ deals_raw_nov25.tsv $(pwd)/fundraising.tsv
```

To run the notebook:
```
uv run jupyter notebook
```

To run model training on a remote GPU [using Modal](https://modal.com/docs/guide#getting-started):
```
uv run python -m modal setup
uv run modal run fundraising.py
```
