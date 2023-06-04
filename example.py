from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (cross_validate, StratifiedKFold)
from sklearn.datasets import load_iris

import smote_variants as sv

dataset = load_iris()
X = dataset['data']
y = dataset['target']

y[y > 1] = 1

oversampler = ('smote_variants', 'ADASYN', {'proportion': 1.0})

classifier = ('sklearn.neighbors', 'KNeighborsClassifier', {})

model= Pipeline([('scale', StandardScaler()),
('clf', sv.classifiers.OversamplingClassifier(oversampler, classifier))])
scoring = ['accuracy','precision', 'recall','f1','roc_auc']
scores = cross_validate(model,X,y,cv=StratifiedKFold(n_splits=10),scoring='roc_auc',error_score='raise')


print(scores)
