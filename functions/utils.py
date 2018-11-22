import os

def checkFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

