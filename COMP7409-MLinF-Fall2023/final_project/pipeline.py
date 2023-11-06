import numpy as np
from colorama import init, Fore, Style
init(autoreset=True)

class Pipeline:
    """
    This class will use the Dataset and SimpleSVM classes to train the SVM model and make predictions.
    """
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def run(self, weights_save_file):
        print(f"{Fore.GREEN}Running pipeline with model {self.model} and dataset {self.dataset.file_path}.{Style.RESET_ALL}")
        # Train the model
        X_train, y_train = self.dataset.get_train_data()
        print(f"{Fore.YELLOW}Training...{Style.RESET_ALL}")
        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            # TODO add reflection or support user to modify the wrong code.
            pass
        print(f"{Fore.GREEN}Finish training.{Style.RESET_ALL}")
        self.model.save_params(weights_save_file)

        # Make predictions
        X_test, _ = self.dataset.get_test_data()
        print(f"{Fore.YELLOW}Predicting...{Style.RESET_ALL}")
        predictions = self.model.predict(X_test)
        print(f"{Fore.GREEN}Finish predicting.{Style.RESET_ALL}")
        return predictions

    