# SyntheticContradetect Dataset

## Overview
This is a synthetically generated contradiction detection dataset designed to cover textual sentence-pair contradictions over a wide array of subjects, sentence structures, and contradiction types. Samples are labeled with 5 main columns: Premise, Hypothesis, Relationship, ContradictionType, and Subject. 

## Intended Use
SyntheticContradetect is designed for research on contradiction detection, natural language inference, and robustness testing.

## Limitations
- Due to being a synthetically generated dataset, many samples do not use proper nouns, and may not capture real-world linguistic ambiguity
- Imbalance in number of each Relationship label
- Imbalance in number of each ContradictionType label
- Imbalance in number of each Subject label
- Text based sentence pair only
- Not intended for safety-critical applications

## Authors
- Jamie Keene  
- Dr. Thejas Gubbi Sadashiva

## Structure
```
main/
├── SyntheticContradetect.xlsx        # Full dataset (2,581 rows)
├──SyntheticContradetect Analysis.ipynb
├──SyntheticContradetect Basic Experiment Code.py
├──SyntheticContradetect Transformer Experiment Code.ipynb
```


## Data Schema
| Column                      | Type   | Description                                                             |
|-----------------------------|--------|-------------------------------------------------------------------------|
| Premise                     | String | Sentence #1                                                             |
| Hypothesis                  | String | Sentence #2                                                             |
| Relationship                | String | Contradictory, Entailing, or Neutral                                    |
| ContradictionType           | String | The category of contradiction for contradicting samples                 |
| Subject                     | String | The category of content the sentences fall into                         |


## Usage

```python

import pandas as pd
df = pd.read_excel("SyntheticContradetect.xlsx")
print(df.head())  # printing first 4 rows

```

For full preprocessing steps, model training, and evaluation results, see `analysis.ipynb`

## Example
Premise: "The batter hit a grand slam in the bottom of the ninth to win the game."

Hypothesis: "The batter struck out with the bases loaded to end the game."

Relationship: Contradictory

ContradictionType: Causal

Subject: Baseball

## How To Cite
Jamie Keene, Thejas G.S., "Human Verified AI-Generated Sentence-Pair Dataset for Contradiction Detection Across Multiple Topics and Contradiction Types", In proceedings of the 2026 IEEE 8th International Conference on Cybernetics, Cognition \& Machine Learning Applications (ICCCMLA), Germany, Oct 5-6, 2026. 
  
@misc{syntheticcontradetect2026,
  title         = {SyntheticContradetect},
  author        = {Keene, Jamie and Gubbi Sadashiva, Thejas},
  year          = {2026},
  howpublished  = {\url{https://github.com/thejasgs/SyntheticContradetect}},
  note          = {Dataset},
  version	= {v1.0}
}

## License
This dataset is licensed under the Creative Commons Attribution–ShareAlike 4.0 International License (CC BY-SA 4.0).

You are free to:
- Share — copy and redistribute the material in any medium or format  
- Adapt — remix, transform, and build upon the material  

Under the following terms:
- Attribution — You must give appropriate credit.  
- ShareAlike — If you remix or build upon the dataset, you must distribute your contributions under the same license.

For full license details, see: https://creativecommons.org/licenses/by-sa/4.0/

![version](https://img.shields.io/badge/version-1.0-blue)


