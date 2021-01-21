## LowFER
Code for the paper "LowFER: Low-rank Bilinear Pooling for Link Prediction", ICML 2020

**NOTE**: The code is based on the open-source code released by TuckER. Please refer to [this](https://github.com/ibalazevic/TuckER) link for the original code and citing their work.

### Experiments
See `experiments.txt` for the commands to run the model with best hyperparameters.

### Requirements
The original codebase was implemented in Python 3.6.6. Required packages are:

    numpy      1.15.1
    pytorch    1.0.1

### Scripts
Run the scripts from main directory as `python -m scripts.filename`:

`toy_example.py`: This script contains the toy dataset used to visualize the Proposition 1 in paper. It runs LowFER under the conditions specified and shows that it perfectly separates positive examples from negative examples.

`bilinear_models_relations.py`: This script runs the conditions presented in relations with bilinear models (section 4.3) with toy setup and show equivalence between LowFER version of other bilinear models and the true scoring functions of those models.

`relation_results_analysis.py`: Evaluates per relation metrics on WN18/RR dataset using LowFER as reported in the per relation analysis results (section 5.4). Note, it requires a trained model so please first run LowFER for WN18/RR as detailed in `experiments.txt`.

`plots.py`: Simple plots script used to generate effect of `k` and `de`.

**Update**: A refactored version of the code can be found on the `refactor` branch with faster evaluation. The results there are slightly less (in 3rd decimal) than this implementation.

### Citation
```
@inproceedings{amin2020lowfer,
  title={LowFER: Low-rank Bilinear Pooling for Link Prediction},
  author={Amin, Saadullah and Varanasi, Stalin and Dunfield, Katherine Ann and Neumann, G{\"u}nter},
  booktitle={International Conference on Machine Learning},
  pages={257--268},
  year={2020},
  organization={PMLR}
}
```
