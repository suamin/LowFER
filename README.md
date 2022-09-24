# LowFER

Code for the paper "LowFER: Low-rank Bilinear Pooling for Link Prediction", ICML 2020

### Experiments

See `experiments.txt` for the commands to run the model with the best hyperparameters.

### Requirements

The original codebase was implemented in Python 3.6.6. Required packages are:
```
numpy      1.15.1
pytorch    1.0.1
```
### Scripts

Run the scripts from main directory as `python -m scripts.filename`:

`toy_example.py`: This script contains the toy dataset used to visualize Proposition 1 in the paper. It runs LowFER under the conditions specified and shows that it perfectly separates positive examples from negative examples.

`bilinear_models_relations.py`: This script runs the conditions presented in relation to bilinear models (section 4.3) with a toy setup and shows the equivalence between the LowFER version of other bilinear models and the true scoring functions of those models.

`relation_results_analysis.py`: Evaluates per relation metrics on WN18/RR dataset using LowFER as reported in the per relation analysis results (section 5.4). It requires a trained model, so please first run LowFER for WN18/RR as detailed in `experiments.txt`.

`plots.py`: Simple plots script used to generate the effect of `k` and `de`.

**Update**: A refactored code version can be found on the `refactor` branch with faster evaluation. The results are slightly less (in third decimal) than this implementation.

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{pmlr-v119-amin20a,
  title ={{L}ow{FER}: Low-rank Bilinear Pooling for Link Prediction},
  author = {Amin, Saadullah and Varanasi, Stalin and Dunfield, Katherine Ann and Neumann, G{\"u}nter},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  pages = {257--268},
  year = {2020},
  editor = {III, Hal Daumé and Singh, Aarti},
  volume = {119},
  series = {Proceedings of Machine Learning Research},
  month = {13--18 Jul},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v119/amin20a/amin20a.pdf},
  url = {https://proceedings.mlr.press/v119/amin20a.html}
}
```

Also, check our follow-up work extending LowFER with time-aware and model-agnostic temporal representations for TKGC and the accompanying temporal knowledge graph embeddings framework [ChronoKGE](https://github.com/iodike/ChronoKGE):

```bibtex
@inproceedings{dikeoulias-etal-2022-temporal,
    title = "Temporal Knowledge Graph Reasoning with Low-rank and Model-agnostic Representations",
    author = {Dikeoulias, Ioannis and Amin, Saadullah and Neumann, G{\"u}nter},
    booktitle = "Proceedings of the 7th Workshop on Representation Learning for NLP",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.repl4nlp-1.12",
    doi = "10.18653/v1/2022.repl4nlp-1.12",
    pages = "111--120",
}
```

## Acknowledgements

The code is based on the open-source code released by TuckER. If you find our work useful, please consider [this](https://github.com/ibalazevic/TuckER) link for the original code and cite their work.
