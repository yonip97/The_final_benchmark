# Final Benchmark

**The Final Benchmark** is a benchmark designed to evaluate the ability of LLMs to perform fine grained detection of factual inconsistencies in summaries.  

For more details, please see our paper: TODO: put link to paper.

---

## Dataset

The dataset is located in the `data` directory and is divided into **development** and **test** sets.  

Each entry in the dataset contains the following fields:

1. **text** – The original XSum text.  
2. **summary** – A Pegasus generated summary of the text.  
3. **human_descriptions** – A list of human annotations, where each entry provides a natural language description of a single factual inconsistency in the summary. 
4. **set** – Indicates whether the entry belongs to the `dev` or `test` split.  
5. **DeFacto_label** – The original DeFacto label: either `consistent` or `inconsistent`.  

---

## Evaluation Pipeline

**TODO**: 