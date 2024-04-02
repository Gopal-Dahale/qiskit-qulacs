"""Utility functions for tests"""

from unittest.util import safe_repr


def dicts_almost_equal(dict1, dict2, delta=None, places=None, default_value=0):
    """Test if two dictionaries with numeric values are almost equal.

    Fail if the two dictionaries are unequal as determined by
    comparing that the difference between values with the same key are
    not greater than delta (default 1e-8), or that difference rounded
    to the given number of decimal places is not zero. If a key in one
    dictionary is not in the other the default_value keyword argument
    will be used for the missing value (default 0). If the two objects
    compare equal then they will automatically compare almost equal.

    Args:
      dict1 (dict): a dictionary.
      dict2 (dict): a dictionary.
      delta (number): threshold for comparison (defaults to 1e-8).
      places (int): number of decimal places for comparison.
      default_value (number): default value for missing keys.

    Raises:
          TypeError: if the arguments are not valid (both `delta` and
        `places` are specified).

    Returns:
          String: Empty string if dictionaries are almost equal. A description
        of their difference if they are deemed not almost equal.
    """

    def valid_comparison(value):
        """compare value to delta, within places accuracy"""
        if places is not None:
            return round(value, places) == 0
        else:
            return value < delta

    # Check arguments.
    if dict1 == dict2:
        return ""
    if places is not None:
        if delta is not None:
            raise TypeError("specify delta or places not both")
        msg_suffix = f" within {places} places"
    else:
        delta = delta or 1e-8
        msg_suffix = f" within {delta} delta"

    # Compare all keys in both dicts, populating error_msg.
    error_msg = ""
    for key in set(dict1.keys()) | set(dict2.keys()):
        val1 = dict1.get(key, default_value)
        val2 = dict2.get(key, default_value)
        if not valid_comparison(abs(val1 - val2)):
            error_msg += f"({safe_repr(key)}: {safe_repr(val1)} != {safe_repr(val2)}), "

    if error_msg:
        return error_msg[:-2] + msg_suffix
    else:
        return ""
