# -*- coding: utf-8 -*-

from box import Box

from el_config import ConfigBase

from .helpers import defaults
from .helpers.validators import _config_schema


def _pre_load(config):
    _config = config.copy()
    config.tsc_vrae = Box(defaults.tsc_vrae)
    config.merge_update(_config)
    return config


_cb = ConfigBase(pre_load=_pre_load, valid_schema=_config_schema)
_cb.load()
config = _cb.config
