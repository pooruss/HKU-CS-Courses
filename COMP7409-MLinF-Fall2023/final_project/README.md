The test folder is irrelevant to the project. It only contains some code used in the report to show the performance of some algorithms. The user of this project do not need to open it and the python file in the test folder will not affect the main project.

After that, all you need is to run main.py. First, the program will let you to input the dataset by showing “please input a dataset(the input form should be: xxx.csv):”. We have 6 datasets for users to choose, so you can type `boston_house_prices.csv` or `Customers.csv` or `iris.csv` or `fashion-mnist.csv` or `mobile_price.csv` or `iris_b.csv”`to select the dataset. Then, it will show features,the target and first five rows of data for you to let you have a brief view of the dataset. It can help you to select the proper preprocessing method. Then the program will show “please preprocess the dataset(we have normalization and standardization, you can also type no to choose neither of them)”. The user can type normalization or standardization or no, to use one of the preprocessing methods or not to use any of them. If you type no, the program will show “it seems you do not need to preprocess the data, that is ok”. After that, it will provide you several machine learning algorithms which can be used to this dataset. For example, if I choose to use iris dataset(which means you type iris.csv before), it will show you:

“You can select these ML approaches by using iris dataset: Classification: Naive Bayes, Perceptron, Multilayer Perceptron, Iterative Dichotomiser 3, C4.5. Unsupervised: K-Means Clustering, 'Principal Component Analysis, 'Density-Based Spatial Clustering of Applications with Noise.”

Now, you need to type the name of the machine learning like Naive Bayes to select the algorithm you want to use. After a while, the program will show the evaluation result. Different algorithms use different evaluation methods so the result is different. The user can see the performance of the selected algorithm of the selected dataset.





Note: 

- Native Bayes, ID3, Decision Tree, and C4.5 can not use `fashion-mnist.csv` dataset.
- ID3, Decision Tree can not use `mobile_price.csv` dataset
- Multilayer Perceptron (MLP) and Linear Regression **must** normalize the dataset whatever you choose.