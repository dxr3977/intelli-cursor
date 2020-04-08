from cursor_training import *
from cursor_data_acquisitioner import *
import os



def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def main():
    print("CWD = " + os.path.dirname(os.path.realpath(__file__)))
    path = os.path.dirname(os.path.realpath(__file__))
    print("\n\nlist files ..")
    print(list_files(os.path.join(path, "../")))

    print("\n\nlist files ../..")
    print(list_files(os.path.join(path, "../../")))

    print("it works")
    import pyautogui as p
    while True:
        print(p.position())

main()