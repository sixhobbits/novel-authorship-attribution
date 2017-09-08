from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from kerashelper import *


def test_hello():
    # arrange 
    expected = "hello"

    # act
    actual = hello()

    # assert
    return expected == actual


def test_hello_2():
    # arrange 
    not_expected = "goodbye"

    # act 
    actual = hello()

    # assert
    return not_expected != actual
