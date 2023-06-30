# Clinical Presence Fairness
This repository allows to reproduce results from the paper [Imputation Strategies Under Clinical Presence: Impact on Algorithmic Fairness](https://arxiv.org/abs/2208.06648). This paper explores the importance of imputation on algorithmic fairness.

## How to reproduce the paper main findings ?
### Synthetic
The folder `synthetic` contains all the functions and experiments for reproducing the simulation experiments. The notebook allows to enforce the clinical presence patterns identified in the paper. The following figure introduces these different scenarios of clinical presence, i.e. the complex interaction between patients and healthcare, that can result in group-specific missingness patterns.

![Model](./images/scenarios.png)

### MIMIC
The folder `mimic` contains three notebooks. First, run `preprocessing.ipynb` to extract the labratory tests and the study population. Then, `experiment.ipynb` to run the different imputation pipelines. Finally, `analysis_group.ipynb` compare the pipeline performances.
## Findings
- **Insight 3.0** - Real-world data presents group-specific clinical missingness. 
- **Insight 3.1** - Different imputation strategies may have similar prediction performance at the population level while having \textbf{opposite} group performance gaps.  
- **Insight 3.2** - No imputation strategy consistently outperforms the others across groups. 
- **Insight 3.3** - Current recommendations for group-specific imputation and use of missingness indicators can increase the performance gap and yield a worse performance for the marginalised groups.  

## Future directions
- Quantifying risk.
- Clinical presence can result in group-specific temporal patterns that we would like to explore.
## Requirements
This paper relies on `skcikit-learn`, `matplotlib` and `seaborn`. For reproducing the MIMIC III results, access to the dataset needs to be granted. 