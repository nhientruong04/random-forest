from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from scipy.stats import mode

class RandomForestClassifier:
    """The model assume oob score is used."""

    def __init__(self, n_estimators=100):
        self.T = n_estimators
        self.oob_score = 0

        self.trees_list = list()
        self.oob_list = list()

    def __bootstrap_dataset(self, X, Y):
        assert X.shape[0] == Y.shape[0], "Different sample num"

        sample_indices = np.random.choice(len(X), len(X))
        oob_indices = np.setdiff1d(np.arange(X.shape[0]), sample_indices)

        return {
            "train_set": (X[sample_indices], Y[sample_indices]),
            "oob_set": (X[oob_indices], Y[oob_indices]),
            "oob_indices": oob_indices
        }
    
    def predict(self, X):
        outputs = list()

        for tree_t in self.trees_list:
            outputs.append(tree_t.predict(X))

        outputs = np.array(outputs) # shape (n_trees, n_samples)
        
        return mode(outputs, axis=0).mode
    
    def fit(self, X, Y):
        # empty previous run
        self.trees_list = list()
        self.oob_list = [list() for _ in range(X.shape[0])]

        for t in range(self.T):
            # get bootstrapped and oob dataset for this tree
            bootstrapped_ds = self.__bootstrap_dataset(X, Y) 
            train_X, train_Y = bootstrapped_ds["train_set"]
            oob_X, oob_Y = bootstrapped_ds["oob_set"]
            oob_indices = bootstrapped_ds["oob_indices"]

            # train
            tree_t = DecisionTreeClassifier(max_features="sqrt", random_state=0)
            tree_t.fit(train_X, train_Y)
            self.trees_list.append(tree_t)

            # calculate oob accuracy for this tree
            oob_preds = list(tree_t.predict(oob_X))

            for i, oob_pred in enumerate(oob_preds):
                self.oob_list[oob_indices[i]].append(oob_pred)

        aggregated_oob_preds = np.array([mode(sample_pred).mode for sample_pred in self.oob_list])
        final_oob_indices = ~np.isnan(aggregated_oob_preds)
        self.oob_score = accuracy_score(aggregated_oob_preds[final_oob_indices], Y[final_oob_indices])