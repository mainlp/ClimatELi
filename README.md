# ClimatELi

This repo contains the data and code introduced in the following ClimateNLP 2024 workshop paper @ ACL 2024:

[CLIMATELI: Evaluating Entity Linking on Climate Change Data](https://arxiv.org/abs/2406.16732)

## Contributions
- We present CLIMATELI (CLIMATe Entity LInking), the first manually annotated CC dataset that links 3,087 entity spans to Wikipedia. 
- We evaluate existing entity linking (EL) systems on the CC topic across various genres and propose automated filtering methods for CC entities.
- We find that the performance of EL models notably lags behind humans at both token and entity levels. Testing within the scope of retaining or excluding non-nominal and/or non-CC entities particularly impacts the models’ performances.

## The ClimatELi Dataset
The human annotation for each document is in `dataset/gold_annotation`.

We include ten English documents across five genres — Wikipedia pages, academic articles, web news, United Nations’ Intergovernmental Panel on Climate Change reports, and YouTube transcriptions — consisting of 3,087 (1,161 unique) entity links across 12,802 tokens.

The data is in the tsv format:

| Token        | POS   | Label       | Category           |
|--------------|-------|-------------|--------------------|
| A            | DET   | O           |                    |
| Relationship | NOUN  | O           |                    |
| between      | ADP   | O           |                    |
| Climate      | PROPN | B           | Climate_finance    |
| Finance      | NOUN  | I           | Climate_finance    |
| and          | CCONJ | O           |                    |
| Climate      | NOUN  | B           | Climate_risk       |
| Risk         | NOUN  | I           | Climate_risk       |
| :            | PUNCT | O           |                    |
| Evidence     | NOUN  | B           | Evidence           |
| from         | ADP   | O           |                    |
| the          | DET   | O           |                    |
| South        | PROPN | B           | South_Asia         |
| Asian        | ADJ   | I           | South_Asia         |
| Region       | PROPN | I           | South_Asia         |

## Code
### Please follow these steps to reproduce results in the paper:
- Pip install required packages.
```
pip install -r requirements.txt
```
- Generate EL predictions by model
```
cd utils
python3 generator.py
```
- Validate the EL predictions by model with gold annotations
```
python3 validator.py
```
- Evaluate the EL predictions by model
```
python3 evaluator.py
```
- Visualize the evaluation results
```
python3 visualize.py
```



## Reference

### Please cite the following paper:

```bibtex
@inproceedings{
zhou2024climateli,
title={{CLIMATELI}: Evaluating Entity Linking on Climate Change Data},
author={Shijia Zhou and Siyao Peng and Barbara Plank},
booktitle={Natural Language Processing meets Climate Change @ ACL 2024},
year={2024},
url={https://openreview.net/forum?id=TXmTYknzwk}
}
```

## Acknowledgement
- This work belongs to the KLIMA-MEMES project funded by the Bavarian Research Institute for Digital Transformation (bidt), an institute of the Bavarian Academy of Sciences and Humanities.
- The authors are responsible for the content of this publication.