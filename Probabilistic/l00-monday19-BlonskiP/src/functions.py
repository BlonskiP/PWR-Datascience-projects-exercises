"""Tasks for checking knowledge about functions."""
from typing import List


def task_00(fun, x):
    """Checks if student can accept functions as arguments.

    Modify this function signature, so it can accept a function and an integer.
    Call the function with the integer value. Assume the function returns
    a string and add following string to the result: '_END'. Return the whole
    string from this function.
    """
    result = fun(x) + "_END"
    return result


def task_01(nums: List[int]):
    """Checks if student can declare and use nested functions.

    Inside this function declare another function named `absolute`, which
    calculates the absolute value for a given argument. Then use `map` and
    `filter` functions to apply `absolute` on all elements of `nums` and
    remove all values greater or equal to 5.
    """
    def absolute(number):
        if number > 0:
            return number
        else:
            return -number

    result = map(absolute, nums)
    result = filter(lambda x: x < 5, result)
    return list(result)
