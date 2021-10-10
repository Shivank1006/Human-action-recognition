import matplotlib.pyplot as plt 
import os
from matplotlib import style

style.use('fivethirtyeight')

folders = os.listdir('./train_images')
train_number = []
test_number = []

for folder in folders:
    train_files = os.listdir('./train/' + folder)
    test_files = os.listdir('./test/' + folder)

    train_number.append(len(train_files))
    test_number.append(len(test_files))

print(sum(train_number))
print(sum(test_number))

# print(test_number)
# print(train_number)
# plt.bar(folders, train_number)
# plt.xticks(folders, rotation='vertical')
# plt.show()

# plt.bar(folders, test_number)
# plt.xticks(folders, rotation='vertical')
# plt.show()
# print(folders)