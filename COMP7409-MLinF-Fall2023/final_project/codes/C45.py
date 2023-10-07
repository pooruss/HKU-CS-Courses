# 定义节点类
import math


class Node:
    # identify the structure of the decision tree
    def __init__(self, col=-1, value=None, class_label=None, left=None, right=None):
        self.col = col  # Characteristic column index
        self.value = value  # Value of dividing points
        self.class_label = class_label  # Category labels for leaf nodes
        self.left = left  # left subtree
        self.right = right  # right subtree

class C45:

    def __init__(self, traindata1, testdata1):
        self.train_data=traindata1
        self.test_data=testdata1

    """
    Function description: calculate the empirical entropy of a given dataset (Shannon entropy)
    Parameters:
        dataSet 
    Returns:
        entropy
    """
    # Calculate the entropy of the dataset
    def calc_entropy(self, data):
        num_samples = len(data)
        label_counts = {}
        for sample in data:
            label = sample[-1]
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        entropy = 0.0
        for label in label_counts:
            prob = float(label_counts[label]) / num_samples
            entropy -= prob * math.log(prob, 2)
        return entropy

    """
    Function description: calculate the conditional entropy of a given dataset 
    Parameters:
        dataSet, column 
    Returns:
        conditional entropy
    """
    # Calculate conditional entropy
    def calc_cond_entropy(self, data, col):
        num_samples = len(data)
        feature_values = {}
        for sample in data:
            feature_value = sample[col]
            if feature_value not in feature_values:
                feature_values[feature_value] = []
            feature_values[feature_value].append(sample)
        cond_entropy = 0.0
        for feature_value in feature_values:
            prob = float(len(feature_values[feature_value])) / num_samples
            sub_data = feature_values[feature_value]
            cond_entropy += prob * self.calc_entropy(sub_data)
        return cond_entropy

    """
    Function description: calculate the information gain of a given dataset 
    Parameters:
        dataSet, column
    Returns:
        information gain
    """
    # Calculate information gain
    def calc_info_gain(self, data, col):
        entropy = self.calc_entropy(data)
        cond_entropy = self.calc_cond_entropy(data, col)
        return entropy - cond_entropy


    """
    Function description: select the best feature

    Parameters:

    DataSet - Dataset

    Returns:

    BestFeature - Index value of the (optimal) feature with the largest information gain
    """
    # Select the optimal feature
    def choose_best_feature(self, data):
        num_features = len(data[0]) - 1
        best_feature = -1
        best_info_gain = 0.0
        for col in range(num_features):
            info_gain = self.calc_info_gain(data, col)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = col
        return best_feature

    """
    Function description: recursively build a decision tree

    Parameters:

    DataSet - training dataset


    Returns:

    MyTree - Decision Tree
    """
    # Building a Decision Tree
    def build_tree(self, data):
        class_labels = [sample[-1] for sample in data]
        if class_labels.count(class_labels[0]) == len(class_labels):
            # When all category labels are the same, return the leaf node
            return Node(class_label=class_labels[0])
        if len(data[0]) == 1:
            # When only the category label column is left, return the leaf node with the category label being the category with the most occurrences
            class_label = max(set(class_labels), key=class_labels.count)
            return Node(class_label=class_label)
        best_feature = self.choose_best_feature(data)
        node = Node(col=best_feature)
        feature_values = set([sample[best_feature] for sample in data])
        for feature_value in feature_values:
            # Partition Dataset
            sub_data = []
            for sample in data:
                if sample[best_feature] == feature_value:
                    sub_sample = sample[:best_feature] + sample[best_feature + 1:]
                    sub_data.append(sub_sample)
            if len(sub_data) == 0:
                # When the subset is empty, return the leaf node with the category label as the category with the most occurrences
                class_label = max(set(class_labels), key=class_labels.count)
                node.left = node.right = Node(class_label=class_label)
            else:
                # 递归构建子树
                sub_tree = self.build_tree(sub_data)
                if sub_tree.class_label is not None:
                    node.left = node.right = sub_tree
                else:
                    if sub_tree.left is None:
                        # When the left subtree of the subtree is empty, the leaf node is returned, and its category label is the category with the most occurrences
                        class_label = max(set(class_labels), key=class_labels.count)
                        node.left = Node(class_label=class_label)
                    else:
                        node.left = sub_tree.left
                    if sub_tree.right is None:
                        # When the right subtree of the subtree is empty, the leaf node is returned, and its category label is the category with the most occurrences
                        class_label = max(set(class_labels), key=class_labels.count)
                        node.right = Node(class_label=class_label)
                    else:
                        node.right = sub_tree.right
            node.value = feature_value
        return node


    """
    Function description: estimate the type of the sample

    Parameters:

    sample, tree

    Returns:

    tree.classLabel
    """
    # Category of predicted samples
    def predict(self, sample, tree):
        while tree.class_label is None:
            col = tree.col
            feature_value = sample[col]
            if feature_value == tree.value:
                tree = tree.left
            else:
                tree = tree.right
        return tree.class_label




    def c45(self):

        tree = self.build_tree(self.train_data)

        num_correct = 0
        for sample in self.test_data:
            label = self.predict(sample, tree)
            if label == sample[-1]:
                num_correct += 1
        accuracy = float(num_correct) / len(self.test_data)
        print(label)
        print("Accuracy:", accuracy)