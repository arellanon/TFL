# grid search solver for lda
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = LinearDiscriminantAnalysis()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
#rid = dict()
#grid['solver'] = ['lsqr', 'eigen']
#grid['shrinkage'] = np.arange(0, 1, 0.01)


"""
param_lsqr = {
    "solver": ["lsqr", "eigen"],
    "shrinkage": [ None, "auto"] + np.arange(0, 1, 0.01).tolist()
}

param_svd = {
    "solver": ["svd"],
    "store_covariance": [True, False],
    "tol": np.arange(0, 0.0001, 0.0000001)
}

param_grid = [ param_lsqr, param_svd]
"""


param_grid = [
    {
    "solver": ["lsqr", "eigen"],
    "shrinkage": [ None, "auto"] + np.arange(0, 1, 0.01).tolist()
    }, 
    {
    "solver": ["svd"],
    "store_covariance": [True, False],
    "tol": np.arange(0, 0.0001, 0.0000001)
    } ]

# define search
search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
#print(results.cv_results_)

print(sorted(results.cv_results_.keys()))
"""
print(results.cv_results_)
#display(df)
print(df.head(3))
"""
#df = pd.DataFrame(results.cv_results_)

df = pd.DataFrame(results.cv_results_)[
#    ["param_solver", "param_store_covariance",  "param_tol", "mean_test_score"]
    ["param_solver", "param_shrinkage", "param_store_covariance",  "param_tol", "params", "mean_test_score", "std_test_score"]
]
"""
df["mean_test_score"] = -df["mean_test_score"]
df = df.rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
    }
)
df.sort_values(by="BIC score").head()
"""
#display(df)
df.to_csv('raw_data.csv', index=False)