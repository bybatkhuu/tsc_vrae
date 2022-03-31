# -*- coding: utf-8 -*-

import os
import errno
from random import randint

from pydantic import validate_arguments

from el_logging import logger
from el_validator import checkers


@validate_arguments()
def create_dir(create_dir: str):
    """Create directory if 'create_dir' doesn't exists.

    Args:
        create_dir (str): Directory path.
    """

    if not os.path.isdir(create_dir):
        logger.warning(f"'{create_dir}' directory doesn't exist!")
        try:
            os.makedirs(create_dir)
        except OSError as err:
            if err.errno == errno.EEXIST:
                logger.info(f"'{create_dir}' directory already exists!")
            else:
                logger.error(f"Failed to create '{create_dir}' directory.")
                raise
        logger.success(f"Successfully created '{create_dir}' directory!")



@validate_arguments
def clean_obj_dict(obj_dict: dict, cls_name: str):
    """Clean class name from object.__dict__ for str(object).

    Args:
        obj_dict (dict, required): Object dictionary by object.__dict__.
        cls_name (str , required): Class name by cls.__name__.

    Returns:
        dict: Clean object dictionary.
    """

    try:
        if checkers.is_empty(obj_dict):
            raise ValueError("'obj_dict' argument value is empty!")

        if checkers.is_empty(cls_name):
            raise ValueError("'cls_name' argument value is empty!")
    except ValueError as err:
        logger.error(err)
        raise

    _self_dict = obj_dict.copy()
    for _key in _self_dict.keys():
        _class_prefix = f"_{cls_name}__"
        if _key.startswith(_class_prefix):
            _new_key = _key.replace(_class_prefix, '')
            _self_dict[_new_key] = _self_dict.pop(_key)
    return _self_dict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def obj_to_repr(obj: object):
    """Modifying object default repr() to custom info.

    Args:
        obj (object, required): Any python object.

    Returns:
        str: String for repr() method.
    """

    try:
        if checkers.is_empty(obj):
            raise ValueError("'obj' argument value is empty!")
    except ValueError as err:
        logger.error(err)
        raise

    return f"<{obj.__class__.__module__}.{obj.__class__.__name__} object at {hex(id(obj))}: " + "{" + f"{str(dir(obj)).replace('[', '').replace(']', '')}" + "}>"


@validate_arguments
def rand_color(start: int=10, end: int=230):
    """Generate random color. Avoid to too light or too dark colors.

    Args:
        start (int, optional): Starting number for each RGB code. Defaults to 10.
        end   (int, optional): Ending number for each RGB code. Defaults to 230.

    Raises:
        ValueError: If 'start' or 'end' argument value is not int.
        ValueError: If 'start' or 'end' argument value is less than 0 and greater than 255.

    Returns:
        str: Hex color as a string. Example: '#EE4455'.
    """

    try:
        if not checkers.is_integer(start, minimum=0, maximum=255):
            raise ValueError(f"'start' argument value '{start}' is invalid, it must be an integer between 0 and 255!")

        if not checkers.is_integer(end, minimum=0, maximum=255):
            raise ValueError(f"'end' argument value '{end}' is invalid, it must be an integer between 0 and 255!")
    except ValueError as err:
        logger.error(err)
        raise

    _rand_color = f"#{randint(start, end):02X}{randint(start, end):02X}{randint(start, end):02X}"
    return _rand_color
