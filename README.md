# Clinical Presence Fairness
This repository allows to reproduce results from the paper [Imputation Strategies Under Clinical Presence: Impact on Algorithmic Fairness](https://arxiv.org/abs/2208.06648). This paper explores the importance of imputation on algorithmic fairness.

## How to reproduce the paper main findings ?
### Synthetic
The folder `synthetic` contains all the functions and experiments for reproducing the simulation experiments. The notebook allows to enforce the clinical presence patterns identified in the paper. The following figure introduces these different scenarios of clinical presence, i.e. the complex interaction between patients and healthcare, that can result in group-specific missingness patterns.

![Model](./images/scenarios.png)

### MIMIC
The folder `mimic` contains three notebooks. First, run `preprocessing.ipynb` to extract the labratory tests and the study population. Then, `experiment.ipynb` to run the different imputation pipelines. Finally, `analysis_group.ipynb` compare the pipeline performances.
## Findings
- **Insight 1** - Equally-performing imputation strategies at the population level result in different marginalised group performances.  
- **Insight 2** - No strategy consistently outperforms the others across clinical presence scenarios.  
- **Insight 3** - Current recommendation of leveraging additional covariates to make MAR assumption more plausible can harm marginalised group's performance. 
- **Insight 4** - Real-world data presents group-specific clinical presence patterns.  
- **Insight 5** - Marginalised groups can benefit or be harmed by equally performing imputation strategies at the population level.  
- **Insight 6** - Different marginalised groups may be impacted contrarily by the same imputation strategy.

## Future directions
- Quantifying risk.
- Clinical presence can result in group-specific temporal patterns that we would like to explore.
## Requirements
This paper relies on `skcikit-learn`, `matplotlib` and `seaborn`. For reproducing the MIMIC III results, access to the dataset needs to be granted. 