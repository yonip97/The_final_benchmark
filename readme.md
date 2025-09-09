# The Final Benchmark

This repository contains the dataset and evaluation pipeline accompanying our paper.  


## Dataset

The dataset is located in the `data` directory and is divided into **development** and **test** sets.  

Each entry in the dataset contains the following fields:

1. **text** – The original XSum text.  
2. **summary** – A Pegasus generated summary of the text.  
3. **human_descriptions** – A list of human annotations, where each entry provides a natural language description of a single factual inconsistency in the summary. 
4. **set** – Indicates whether the entry belongs to the `dev` or `test` split.  
5. **DeFacto_label** – The original DeFacto label: either `consistent` or `inconsistent`.  

**Example entry:**

```json
{       
  "text": "Zookeepers say it is very unusual to see a koala with fur this light and with eyes and skin remaining the usual brown black.\ But said the koala's white coat is not caused by a condition called albinism, which is when a human or animal doesn't have the chemical in its hair, skin and eyes that creates colour. Australia Zoo have now asked the public to help name the rare koala. Suggestions so far include Snowflake, Diamond, Pearl and Djendaladi, meaning \"white-haired\" in the Noongar language.",
  "summary": "A rare white koala has been born at Australia Zoo in Perth.",
  "human_descriptions": [
    "The source text doesn't state that the zoo is in Perth.",
    "The summary states that the white koala has been born, but the source text does not mention the koala being born, only that it has a rare white coat."
        ],
  "set": "dev",
  "DeFacto_label": "inconsistent"
    }
 ```
## Evaluation Pipeline

**TODO**: 