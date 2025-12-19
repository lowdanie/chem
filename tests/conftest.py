import jax
import pytest


def pytest_configure(config):
    jax.config.update("jax_enable_x64", True)
