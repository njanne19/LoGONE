import csv

train_labels = []
test_labels = []
with open('labels.csv', newline='\n') as csvfile:
    labelreader = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in labelreader:
        new_row = None
        if i == 0:
            row_len = len(row)
            train_labels.append(row)
            test_labels.append(row)
        elif len(row) != row_len:
            new_row = row[:5] + row[7:]
            if new_row is not None:
                assert len(new_row) == 11
                if i < 4001:
                    train_labels.append(new_row)
                else:
                    test_labels.append(new_row)
        i += 1
print('Training Dataset len: ', len(train_labels))
print('Test Dataset len: ', len(test_labels))
with open("labels_train.csv", 'w', newline="\n") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    for row in train_labels:
        writer.writerow(row)
with open("labels_test.csv", 'w', newline="\n") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    for row in test_labels:
        writer.writerow(row)