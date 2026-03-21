# The Final Benchmark

This repository contains the dataset and evaluation pipeline accompanying the paper: *Fine-Grained Detection of Context-Grounded Hallucinations Using LLMs*.

We introduce **Final**: a benchmark designed to evaluate LLMs’ ability to perform fine-grained localization of factual inconsistencies in a grounded generation setup. We also propose an LLM-based evaluation pipeline to assess model performance on this task.

## Dataset

The data files are in **`data/`** (e.g. **`data/data.jsonl`**). A loader is provided in **`dataloader.py`**.

- **`data/prompts/inference_prompts/<name>/`** — inference prompts: `prompt.txt`, optional `past_text_prompt.txt`
- **`data/prompts/judgement_prompts/<name>/`** — judge prompts (same layout)

The dataset contains 1,405 text–summary pairs, of which 1,121 summaries contain inconsistencies, and a total of 2,131 annotated inconsistencies.

We built the dataset based on the DeFacto dataset introduced in the paper [On Improving Summarization Factual Consistency from Natural Language Feedback](https://arxiv.org/pdf/2212.09968).

Each entry in the dataset contains the following fields:

1. **text** — The original XSum text.  
2. **summary** — A Pegasus generated summary of the text.  
3. **human_descriptions** — A list of human annotations, where each entry provides a natural language description of a single factual inconsistency in the summary.  
4. **split** — `dev` or `test`.  
5. **DeFacto_label** — The original DeFacto label: `consistent` or `inconsistent`.  
6. **doc_id** — DeFacto dataset doc id.

**Example entry:**

```json
{
  "text": "The man hid himself in the rear wheel compartment of the plane which landed at Heathrow Airport on Sunday. He was taken into police custody in London but later released without charge. He had bruises and hypothermia from outside temperatures as low as -41C, Austrian media reported. He survived because the plane flew at a low altitude to avoid stormy weather. The man apparently got under a fence at Schwechat airport in Vienna and climbed into the undercarriage of the first plane he saw without knowing its destination. The plane belonged to a sheikh from the United Arab Emirates and had been standing empty on the tarmac at Schwechat airport since Thursday. It flew without passengers to Heathrow, where the Romanian was picked up by police and arrested for stowing away. He could have been charged or fined or given a fixed penalty, the Metropolitan Police told the BBC. But he was cautioned and freed with no further action being taken, PA news agency reported. The man could also have been handed to the UK Border Agency. But it is understood that there is no immigration issue and that the agency will not seek to deport him, according to PA. As Romania is part of the EU, the man is free to enter the UK. A spokesman for the Civil Aviation Authority (CAA) said the stowaway was \"very lucky\" to be alive. \"If they don't find the right part to stow away, they can be crushed when the undercarriage comes up,\" he said. He added: \"Because of the altitude and temperatures during the flight, there is a severe risk to them through exposure and lack of oxygen. \"If that doesn't kill them, then they could be unconscious when the aircraft descends, and that can mean that when the undercarriage opens again, they will fall out.\" According to Austrian media reports, the man just wanted to get out of Vienna and look for work. Romania is a member of the European Union, so Romanians can travel to the UK for holidays. However, controls on Romanians working in Britain remain in place.",
  "summary": "A 23-year-old Romanian stowaway who survived a six-hour flight from Vienna to London has been released by police.",
  "human_descriptions": [
    "The summary makes up his age",
    "The summary makes up the flight duration"
  ],
  "split": "dev",
  "DeFacto_label": "inconsistent",
  "doc_id": 3022
}
```

---

## How to run

With the conda env activated and `credentials.env` in place, from the repository root:

```bash
python run.py \
  --judged_model gpt-4o-2024-11-20 \
  --judge_model gpt-4o-2024-11-20 \
  --inference_prompt zero_shot1 \
  --judgement_prompt cot_judgement_prompt \
  --split dev
```

Required: `--judged_model`, `--judge_model`, `--inference_prompt`, `--judgement_prompt`. The two prompt arguments are **directory names** under `data/prompts/inference_prompts/` and `data/prompts/judgement_prompts/`.

Common options: `--results_dir` (default `./results`), `--split` (`dev` or `test`), `--inference_workers`, `--judgment_workers`, `--inference_delimiter`, `--judgment_delimiter`, `--allow_duplicates`, `--temperature`, `--max_new_tokens`, `--reasoning-level`, `--api-retries`.

**Model IDs** (see `run.py` → `get_model`): vendor APIs (`openai:...`, `anthropic:...`, `google:...`, or prefixes like `gpt-...`, `claude-...`, `gemini-...`); local Hugging Face with `hf:Org/model` or any id containing `/`.

### Local (on-prem) runs (TBD)

Documentation for local runs is **TBD** (not tested yet).

---

## Environment (Conda)

The conda environment name is **`final-benchmark-env`**. This repo ships two related files in the root:

- **`env.yaml`** – version constraints; used to create/update an env or to **generate** the lockfile.
- **`conda-lock.yml`** – full locked solve (exact builds and hashes). Install with **conda-lock** for a reproducible env that matches the rest of the team / CI.

Install the `conda-lock` CLI once in some base environment: `conda install -c conda-forge conda-lock` or `pip install conda-lock`.

### Install from the lockfile (recommended)

`conda-lock.yml` is for **linux-64**.

```bash
conda-lock install -n final-benchmark-env conda-lock.yml
conda activate final-benchmark-env
```

### Install from `env.yaml` (floating versions)

```bash
conda env create -f env.yaml    # first time
conda activate final-benchmark-env
```

## Credentials

An example **`credentials.env`** is provided in the repository root (empty key placeholders and comments). Fill in your secrets there; `run.py` loads it via `load_credentials()`. Typical keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `HUGGINGFACE_TOKEN` or `HUGGINGFACE_HUB_TOKEN`. Do not commit real secrets.
