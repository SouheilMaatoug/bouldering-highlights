import random

import pytest


@pytest.fixture(scope="module")
def sample():
    """Return a random integer between 0 and 10."""
    return random.randint(0, 10)
