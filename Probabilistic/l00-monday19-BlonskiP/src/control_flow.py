"""Tasks for checking knowledge about control flow."""
from typing import List, Optional, Union


def task_00(x: int) -> Optional[int]:
    """Checks if student can use simple IF statement.

    If the value of the provided argument is greater than or equal to 2,
    multiple it by 3, add 4 and return the resulting value.
    Otherwise do nothing.
    """
    if x >= 2:
        x = x * 3 + 4
        return x


def task_01(x: List[int]) -> Union[str, int]:
    """Checks if student can use IF-ELIF-ELSE statement.

    If the provided list is empty, return the following string: 'EMPTY'.
    If the list contains at least 1 element and the first one is greater
    than 2, then return it multiplied by 3.
    Otherwise return the length of the list.
    """
    if len(x) >= 1 and x[0] > 2:
        return x[0] * 3
    elif x:
        return len(x)
    else:
        return 'EMPTY'
