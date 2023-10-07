# This function is used to deliver ML approaches to the user according to their input datasets.
# input: datasets name
# output: alternative ML approaches

class datasets_to_approaches:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name[0:-4]

    def datasets_to_approach(self):
        target_categorys = []    # store the category of dataset and ML approach
        target_approaches = {}   # store the ML approaches which will be return

        mlApproach = {
            'Regression': ['Linear Regression'],
            'Classification': ['Naive Bayes', 'Multilayer Perceptron', 'Iterative Dichotomiser 3', 'C4.5', 'K-Nearest Neighbor'],
            'Unsupervised': ['K-Means Clustering', 'Principal Component Analysis', 'Density-Based Spatial Clustering of Applications with Noise'],
            'Binary classification':['Supporting Vector Machine', 'Adaboost']
        }

        dataset = {
            'Regression': ['boston_house_prices', 'Customers'],
            'Classification': ['iris', 'fashion-mnist', 'mobile_price'],
            'Unsupervised': ['iris', 'fashion-mnist', 'mobile_price'],
            'Binary classification':['iris_b']
        }

        # distinguish what category of ML approach the dataset can use for
        
        for key, values in dataset.items():
            for value in values:
                if self.dataset_name == value:
                    target_categorys.append(key)

        for key, values in mlApproach.items():
            for category in target_categorys:
                if category == key:
                    trick_approach = []
                    for value in values:
                        trick_approach.append(value)
                    target_approaches[category] = trick_approach

        return target_approaches

    def print_approach(self, approach):
        if len(approach) == 0:
            print("Your dataset is not suitable, please enter another one")
        else:
            print(f"You can select these ML approaches by using {self.dataset_name} dataset: ")
            for key, values in approach.items():
                print(f"{key}:", end=' ')
                for value in values:
                    if value == values[-1]:
                        print(value, end='.')
                    else:
                        print(value, end=', ')
                print()


# dataset_name = "IRIS"     # 这块可以写成一个input，来让用户输入数据库的名称
# test = datasets_to_approaches(dataset_name)
# approaches = test.datasets_to_approach()
# test.print_approach(approaches)
