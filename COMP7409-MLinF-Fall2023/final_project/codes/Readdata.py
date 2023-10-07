import numpy as np
import csv
import os


class Readdata:
    # find the position of dataset
    os.chdir("Datasets")

    data = []
    labeltemp = []

    # build a function to decide whether the item is number
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata  # deal with ASCII code

            unicodedata.numeric(s)  # change the string to float
            return True
        except (TypeError, ValueError):
            pass
            return False

    try:
        input_word = input("please input a dataset(the input form should be: xxx.csv):")

        with open(input_word, "r") as f0:
            df0 = csv.reader(f0)
            df0 = [row for row in df0 if all(row)]
            for col in df0:
                labeltemp.append(col[-1])

            a = labeltemp[1:]
            if is_number(a[0]):
                label = list(map(float, a))
            else:
                label = list(a)

            for row in df0:
                data.append(row)
            dataset = data[1:]

            for i in range(
                len(dataset)
            ):  # this part is used to delete blank and label in dataset
                datarow = dataset[i]
                datarow = datarow[:-1]

                while "" in datarow:
                    datarow.remove("")
                dataset[i] = datarow

    except FileNotFoundError:
        print("The file does not exist, please check the file name")
    except Exception as e:
        print("fail to open the file: ", e)

    # print the first five pieces of data
    print("The first five row of the data:")
    with open(input_word, "r") as file:
        totaldataset = csv.reader(file)
        for i, row in enumerate(totaldataset):
            if i < 6:
                print(list(row))
            else:
                break
    print()

    def to_2d_list(my_list, size):
        # change a one-dimensional list into a two-dimensional list
        my_2d_list = [my_list[i : i + size] for i in range(0, len(my_list), size)]
        return my_2d_list

    # similar to the dictionary, this function is used to transfer words to numbers(use numbers to replace words which used to classify different categories)
    def tansfertodata(num, dataset):
        data_numericnew = []

        col = [row[num] for row in dataset]

        unique_values = list(set(col))

        for i in range(len(col)):
            for j in range(len(unique_values)):
                if col[i] == unique_values[j]:
                    data_numericnew.append(j)
        return data_numericnew

    size = len(dataset[0])  # get the original length of row in the dataset

    new1 = []
    #  traverse every items and change the string which is actually number into number
    for i in range(len(dataset)):
        for item in dataset[i]:
            if isinstance(item, (int, float)):  # if it is a number
                new1.append(item)  # add this into new list
            elif isinstance(item, str):  # if it is a string
                if is_number(item):  # if it is actually a number
                    new1.append(float(item))
                else:
                    new1.append(item)

    new_2d_dataset = to_2d_list(new1, size)

    # eliminate the row which has empty item
    for row in new_2d_dataset:
        if "" in row:
            new_2d_dataset.remove(row)

    # use this list to store which column need to be transfer
    colnum = []

    deciderow = new_2d_dataset[0]
    for i in range(len(deciderow)):
        if isinstance(deciderow[i], str):
            colnum.append(i)

    newrow = []
    # transfer the words to numbers and return the new dataset
    for i in range(len(colnum)):
        newrow.append(tansfertodata(colnum[i], new_2d_dataset))

    for i in range(len(colnum)):
        for j in range(len(new_2d_dataset)):
            new_2d_dataset[j][colnum[i]] = newrow[i][j]

    dataset = new_2d_dataset

    dataset = list(
        zip(*dataset)
    )  # transpose for normalization, and then have to transpose back
    dataset = np.array(dataset).astype(float)

    input_word1 = input(
        "please preprocess the dataset(we have normalization and standardization, you can also type no to choose neither of them):"
    )
    if input_word1.lower() == "normalization":
        for i in range(len(dataset)):
            max_value = np.max(dataset[i])  # get the biggest value
            min_value = np.min(dataset[i])  # get the smallest value
            dataset[i] = (dataset[i] - min_value) / (
                max_value - min_value
            )  # do the normalization
    elif input_word1.lower() == "standardization":
        for i in range(len(dataset)):
            # get the mean and standard deviation
            mean = np.mean(dataset[i])
            std = np.std(dataset[i])

            # do the standardization
            dataset[i] = (dataset[i] - mean) / std
    else:
        print("it seems you do not need to preprocess the data, that is ok")

    dataset = dataset.tolist()
    dataset = list(zip(*dataset))

    train_size = 0.7
    train_dataindex = int(len(dataset) * train_size)
    train_labelindex = int(len(label) * train_size)

    train_data = dataset[:train_dataindex]  # get the train data
    test_data = dataset[train_dataindex:]  # get the test data
    train_label = label[:train_labelindex]  # get the train label
    test_label = label[train_labelindex:]  # get the test label
    # These four data will be used in algorithms, they are all lists, we need to transfer them into np.array using a=np.array(b)
