import shutil

with open('trainlist01.txt') as p:
    mylist = p.read().splitlines()


for i in mylist:
    try:
        m = i.split(sep=' ')
        shutil.copyfile('./UCF50/' + m[0], './train/' + m[0])
    except:
        pass

# import os 
# for i in os.listdir('./UCF50/'):
#     os.mkdir('./train/' + i)