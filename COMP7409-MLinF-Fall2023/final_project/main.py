import argparse
from dataset import BaseDataset
from pipeline import Pipeline
from algorithms import SimpleSVM, AdaBoost

def main(args):
    input_file = args.input_file
    algorithm = args.algorithm
    weights_save_file = args.weights_save_file
    # Instantiate the Dataset
    dataset = BaseDataset(input_file)
    
    # Choose the algorithm
    if algorithm == 'SimpleSVM':
        model = SimpleSVM(args.config_file)
    elif algorithm == 'AdaBoost':
        model = AdaBoost(args.config_file)
    else:
        raise ValueError(f"Algorithm {algorithm} is not supported. Choose 'SimpleSVM' or 'AdaBoost'.")
    
    # Instantiate the Pipeline and run it
    pipeline = Pipeline(dataset, model)
    pipeline.run(weights_save_file=weights_save_file)
    
    # Get the accuracy and print it
    accuracy = pipeline.get_accuracy()
    print(f"The accuracy of {algorithm} on the test set is: {accuracy:.2f}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the SVM or AdaBoost algorithm on a dataset.')
    parser.add_argument('--config_file', type=str, required=True, help='The path to config file.')
    parser.add_argument('--input_file', type=str, required=True, help='The path to the CSV file containing the data.')
    parser.add_argument('--weights_save_file', type=str, required=True, help='The path to save the trained model.')
    parser.add_argument('--algorithm', type=str, required=True, choices=['SimpleSVM', 'AdaBoost'],
                        help='The name of the algorithm to use.')
    
    args = parser.parse_args()
    main(args)