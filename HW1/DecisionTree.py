import numpy as np


class DecisionTree:
    def __init__(self, max_depth, threshold):
        self.max_depth = max_depth
        self.threshold = threshold
        self.used_columns = []
        self.root = None

    def fit(self, X_tr, y_tr):
        self.X_tr = X_tr
        self.y_tr = y_tr.astype(np.int64)
        self.column_types = numeric_or_categorical(self.X_tr)
        self.root = self.train(self.X_tr, self.y_tr, self.max_depth, self.threshold)

    def train(self, X, y, max_depth, threshold, counter=0):

        if counter == max_depth or check_pure_threshold(X, y, threshold):
            leafNode = Node(-1, "leaf", -1)
            leafNode.is_leaf = True
            leafNode.classify_leaf = np.bincount(y).argmax()
            leafNode.children = None
            return leafNode


        counter += 1
        best_decision_column, type_of_column, best_decision_value = decision_column_choose(X, y, self.used_columns, self.column_types)
        self.used_columns.append(best_decision_column)
        ##update used_columns
        if type_of_column == 'categorical':
            splited_data, splited_y, categories = split_categorical_data(X,y, best_decision_column)

        else:
            splited_data, splited_y = split_numeric_data(X,y, best_decision_column, best_decision_value)

        this_node = Node(best_decision_column, type_of_column, best_decision_value)
        for i in range(len(splited_data)):
            child = self.train(splited_data[i], splited_y[i], max_depth, threshold, counter)
            if type_of_column == 'categorical':
                child.category_from_father = categories[i]
            this_node.children.append(child)

        return this_node

    def predict(self, X):
        answers = []
        for i in range(len(X)):

            ans = self.test(X[i],self.root)
            if ans == None:
                answers.append(np.random.randint(2))
            else:
                answers.append(ans)
        return answers

    def test(self, X, root):
        if root.type == 'categorical':
            the_column = root.column

            category = X[the_column]
            for i in range(len(root.children)):

                if root.children[i].category_from_father == category:

                    root = root.children[i]

                    return self.test(X, root)
        elif root.type == 'numeric':
            the_column = root.column
            data_value = X[the_column]
            if data_value <= root.numeric_value:
                return self.test(X, root.children[0])
            else:
                return self.test(X, root.children[1])
        else:
            return root.classify_leaf



def numeric_or_categorical(X):
    columnn_type = []
    column_num = X.shape[1]
    for i in range(column_num):
        if len(np.unique(X[:,i])) <= 5:
            columnn_type.append("categorical")
        else:
            columnn_type.append("numeric")
    return columnn_type


def check_pure_threshold(X, y, threshold):
    """

    @param X:
    @param y:
    @param threshold : number between zero and one:
    @return:
    """

    if ((np.count_nonzero(y == 0))/(y.shape[0])) > threshold or ((np.count_nonzero(y == 1))/(y.shape[0])) > threshold:
        return True
    else:
        return False




def decision_column_choose(X, y, used_columns, column_types):
    column_num, row_num = X.shape[1], X.shape[0]
    best_score = 100000
    best_decision_column = -1
    type_of_column = ""
    best_decision_value = -1


    for i in range(column_num):

        if i not in used_columns:
            if column_types[i] == "numeric":
                unique_numeric_values = np.unique(X[:, i])
                for j in range(len(unique_numeric_values)):

                    current_score = score_numeric(X,y, i, unique_numeric_values[j])
                    if current_score <= best_score:
                        best_score = current_score
                        best_decision_column = i
                        type_of_column = "numeric"
                        best_decision_value = unique_numeric_values[j]

            else:

                current_score = score_categorical(X,y,i)
                if current_score <= best_score:
                    best_score = current_score
                    best_decision_column = i
                    type_of_column = "categorical"
                    best_decision_value = None


    return best_decision_column, type_of_column, best_decision_value


def score_numeric(X, y, split_column, numeric_value):
    splited_data, splited_y = split_numeric_data(X, y, split_column, numeric_value)
    return expected_entropy(splited_data, splited_y)


def split_numeric_data(X, y, split_column, numeric_value):
    splited_data_list = []
    splited_y_list = []
    splited_data_list.append(X[np.where(X[:, split_column] < numeric_value)])
    splited_data_list.append(X[np.where(X[:, split_column] >= numeric_value)])


    splited_y_list.append(y[np.where(X[:, split_column] < numeric_value)])
    splited_y_list.append(y[np.where(X[:, split_column] >= numeric_value)])

    splited_data_list_correct = []
    splited_y_list_correct = []
    for i in range(len(splited_data_list)):
        if len(splited_data_list[i]) != 0:
            splited_data_list_correct.append(splited_data_list[i])
    for i in range(len(splited_y_list)):
        if len(splited_y_list[i]) != 0:
            splited_y_list_correct.append(splited_y_list[i])

    return splited_data_list_correct, splited_y_list_correct


def score_categorical(X, y, split_column):
    splited_data, splited_y, categories = split_categorical_data(X, y, split_column)
    return expected_entropy(splited_data, splited_y)


def split_categorical_data(X, y, split_column):#tested
    categories = np.unique(X[:, split_column])

    splited_data_list = []
    splited_y_list = []
    for i in range(0, len(categories)):

        splited_data_list.append(X[np.where(X[:, split_column] == categories[i])])
        splited_y_list.append(y[np.where(X[:,split_column] == categories[i])])

    splited_data_list_correct = []
    splited_y_list_correct = []
    for i in range(len(splited_data_list)):
        if len(splited_data_list[i]) != 0:
            splited_data_list_correct.append(splited_data_list[i])
    for i in range(len(splited_y_list)):
        if len(splited_y_list[i]) != 0:
            splited_y_list_correct.append(splited_y_list[i])

    return splited_data_list_correct, splited_y_list_correct, categories


def expected_entropy(splitted_data, splited_y):#not tested
    n = 0
    category_count = np.zeros(len(splitted_data))
    for i in range(len(splitted_data)):
        category_count[i] = len(splitted_data[i])
        n += len(splitted_data[i])
    probabilities = category_count / n

    expected_entropy_value = 0
    for i in range(len(splitted_data)):
        expected_entropy_value += probabilities[i] * bernouli_entropy(splited_y[i])
    return expected_entropy_value


def bernouli_entropy(y_branch):
    if y_branch.shape[0] == 0:
        return 0
    one = np.count_nonzero(y_branch == 1) / y_branch.shape[0]
    zero = np.count_nonzero(y_branch == 0) / y_branch.shape[0]
    if not one:
        return -1 * zero * np.log2(zero)
    elif not zero:
        return -1 * one * np.log2(one)
    else:
        return -1 * one * np.log2(one) - zero * np.log2(zero)



class Node:
    def __init__(self, column, type, numeric_value):
        self.column = column
        self.children = []
        self.type = type

        self.numeric_value = numeric_value ##if it is a numeric node

        self.is_leaf = False
        self.classify_leaf = -1 ##if it is a leaf node

        self.category_from_father = -1




