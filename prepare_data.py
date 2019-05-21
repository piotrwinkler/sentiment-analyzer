import csv


def prepare(path):
    """
    This module loads data from source files and divides them in proper way.
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    y = -1
    files = path.split(" ")
    for file in files:
        if ".csv" in file:
            with open(file, 'rt') as f:
                data = csv.reader(f)
                for row in data:
                    review = ""
                    for i in range(len(row)):
                        review = review + row[i]
                    review = review.split("NEGATIVE	")
                    x_test.append(review[1])
                    y_test.append(0)
        elif ".txt" in file:
            f = open(file, "r")
            content = f.read()
            if "NEGATIVE" in content[0:10]:
                content = content[9:]
                content = content.split("\nNEGATIVE	")
                y = 0
            elif "POSITIVE" in content[0:10]:
                content = content[9:]
                content = content.split("\nPOSITIVE	")
                y = 1

            if "test" in file:
                x_test.extend(content)
                y_test.extend([y] * len(content))
            else:
                x_train.extend(content)
                y_train.extend([y] * len(content))

            f.close()
    return x_train, x_test, y_train, y_test
