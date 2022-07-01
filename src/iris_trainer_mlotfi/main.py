from model_trainer import ModelTrainer

from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from hyperopt import hp

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

hyperparam_space = {
   'C': hp.quniform('C', 0.1, 2.0, 0.05)
}
performance_metric = roc_auc_score

trainer = ModelTrainer(LogisticRegression, performance_metric,
                       (X_train, X_test, y_train, y_test), multi_class='ovo')

trainer.add_possible_hyperparameters(hyperparam_space)

result = trainer.run_hyperparameter_search()

print(f'Best hyperparameter found: {result}')
print('Training final model.')

model, final_model_info = trainer.train_final_model()
results_description = 'The final model has'

for i, (k, v) in enumerate(final_model_info.items()):
    if i == len(final_model_info.items()) -1 :
        results_description= f'{results_description} and {k} of {round(v, 2)}'
    else:
        results_description = f'{results_description} {k} of {round(v, 2)},'

results_description += "."

print(results_description)
