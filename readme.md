# The Final Benchmark

This repository contains the dataset and evaluation pipeline accompanying our paper.TODO:ADD PAPER LINK.

Our paper introduces **The Final Benchmark**, a benchmark designed to evaluate LLMs’ ability to perform fine grained detection of factual inconsistencies in summaries. It also proposes an LLM as a judge based evaluation pipeline to assess model performance on this task.

## Dataset

The data files are in **`./data`**, and a simple loader is provided in **``dataloader.py``**. 

Our dataset is adapted from the DeFacto dataset introduced in the paper [On Improving Summarization Factual Consistency from Natural Language Feedback](https://arxiv.org/pdf/2212.09968); we manually curated and adapted DeFacto, then further enriched it via human–LLM collaboration. 
The result is a comprehensive dataset containing 1,405 text–summary pairs, of which 1,121 summaries are inconsistent, with 2,131 annotated inconsistencies.

Each entry in the dataset contains the following fields:

1. **text** – The original XSum text.  
2. **summary** – A Pegasus generated summary of the text.  
3. **human_descriptions** – A list of human annotations, where each entry provides a natural language description of a single factual inconsistency in the summary. 
4. **split** – Indicates whether the entry belongs to the `dev` or `test` split.  
5. **DeFacto_label** – The original DeFacto label: either `consistent` or `inconsistent`.  
6. **doc_id** - DeFacto dataset doc_id.

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
  "doc_id":3022 
    }
 ```
## Evaluation Pipeline

**TODO**: 