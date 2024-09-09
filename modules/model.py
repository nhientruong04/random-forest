from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from scipy.stats import mode

class RandomForestClassifier:
    """The model assume oob score is used."""

    def __init__(self, n_estimators):
        self.T = n_estimators
        self.oob_score = 0

        self.trees_list = list()
        self.oob_list = list()

    def __bootstrap_dataset(self, X, Y):
        assert X.shape[0] == Y.shape[0], "Different sample num"

        sample_indices = np.random.choice(len(X), len(X))
        oob_indices = np.setdiff1d(np.arange(X.shape[0]), sample_indices)

        return X[sample_indices], Y[sample_indices], X[oob_indices], Y[oob_indices]
    
    def predict(self, X):
        outputs = list()

        for tree_t in self.trees_list:
            outputs.append(tree_t.predict(X))

        outputs = np.array(outputs) # shape (n_trees, n_samples)
        
        return mode(outputs, axis=0).mode
    
    def __validate_tree(self, tree, y_true, X):
        outputs = tree.predict(X)

        return accuracy_score(y_true=y_true, y_pred=outputs)

    def fit(self, X, Y):
        # empty previous run
        if len(self.trees_list) > 0:
            self.trees_list = list()
            self.oob_list = list()

        for t in range(self.T):
            # get bootstrapped and oob dataset for this tree
            train_X, train_Y, oob_X, oob_Y = self.__bootstrap_dataset(X, Y) 

            # train
            tree_t = DecisionTreeClassifier(max_features="sqrt", random_state=0)
            tree_t.fit(train_X, train_Y)
            self.trees_list.append(tree_t)

            # calculate oob accuracy for this tree
            oob_accuracy = self.__validate_tree(tree_t, oob_Y, oob_X)
            self.oob_list.append(oob_accuracy)

        self.oob_score = np.mean(np.array(self.oob_list))