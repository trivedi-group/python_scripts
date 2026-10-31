# python_scripts

Python scripts for chemometric analysis of metabolomics and lipidomics data, developed by the Trivedi Research Group at the University of Manchester. They cover data import and pre-processing, unsupervised exploration with principal component analysis (PCA), and supervised classification with discriminant analysis, support vector machines and random forests.

These are working research scripts rather than a packaged library. Most were written for specific datasets and are best used as templates to adapt to your own data.

## What is in the repository

All scripts are in the `MyPython` folder.

### Data import and pre-processing

* `Import Data and Pre Process.py` reads a CSV data matrix, separates features from class labels, imputes missing values with `SimpleImputer`, applies a log₁₀ transform, creates dummy variables for the class label and splits the data into training and test sets.
* `Runtime codes.py` holds short snippets used interactively during analysis.

### Exploratory analysis

* `PCA code.py` performs PCA using the NIPALS algorithm in statsmodels and plots the scores.
* `Correlation Analysis.py` calculates Pearson correlation matrices (with p values) for blocks of features and plots them as heatmaps.
* `Prodromal.py` is a longer exploratory workflow including PCA, alternative imputation and scaling options, and seaborn plots of participant metadata.

### Classification

* `Classification Code.py` produces a PCA scores plot with explained variance on each axis, followed by an RBF kernel SVM classifier and Yellowbrick diagnostic plots.
* `PC-DFA.py` carries out principal component discriminant function analysis, using PCA scores as the input to linear discriminant analysis.
* `Discriminant analysis.py` fits linear discriminant analysis (LDA) to a training set and projects the test set.
* `SVM Classification Code.py` tunes an RBF kernel support vector machine by grid search with five-fold cross-validation, then reports the confusion matrix, classification report and accuracy.
* `PC-SVM function.py` is an early function combining PCA and SVM.
* `Ensemble methods.py` trains a random forest with bootstrap sampling and out-of-bag scoring, then reports performance and feature importances.

### Teaching and test examples

* `SVM.py` plots SVM decision boundaries for different kernels on a two-feature example.
* `plot_confusion_matrix.py` is adapted from the scikit-learn confusion matrix example and uses the Iris dataset.
* `tester file for deletion.py`, `test - to delete.py` and `test.R` are scratch files.

## Requirements

The scripts were written in Python 3.7 and use the following packages.

* pandas
* NumPy
* SciPy
* scikit-learn
* statsmodels
* matplotlib
* seaborn
* Yellowbrick (only for `Classification Code.py`)

Install them with

```
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn yellowbrick
```

Some calls (for example `np.round_` and the `seaborn-poster` style) are deprecated in recent library versions and may need small edits to run.

## Using the scripts

1. Arrange your data as a CSV file with samples in rows, a class label column and feature columns (for example metabolite or lipid intensities).
2. Change the `pd.read_csv()` path at the top of a script to point to your own file. The paths in the repository refer to the original analysis computer and will not work elsewhere.
3. Adjust the `iloc` column ranges so they select your label and feature columns.
4. Run `Import Data and Pre Process.py` first, as several classification scripts expect `X_train`, `X_test`, `y_train` and `y_test` to exist already. Running the scripts in sequence in an interactive console such as IPython or the PyCharm Python console works best.

## Notes

* The `MyPython/venv` folder is a committed Python virtual environment. It is not needed to run the scripts and can be ignored.
* Random seeds are fixed (`random_state=69`) in most scripts so that splits and models are reproducible.
* No data are included in this repository.

## Software references

* Pedregosa, F. et al. (2011) Scikit-learn, machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
* Seabold, S. and Perktold, J. (2010) Statsmodels, econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference*, 92–96.

## Related repositories

* [Chemometrics-in-R](https://github.com/trivedi-group/Chemometrics-in-R), course material for chemometrics in R.
* [papersprayMS](https://github.com/trivedi-group/papersprayMS), R scripts for processing paper spray mass spectrometry data.
