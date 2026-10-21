"""Suite OBS sobre ``SimpleProcessor``."""

import pytest

from vla_learner.policy.processors import SimpleProcessor
from vla_learner.testing.contracts.observation_processor_contract import (
    ObservationProcessorContract,
)

pytestmark = pytest.mark.fast


class TestSimpleProcessor(ObservationProcessorContract):
    @pytest.fixture
    def processor(self) -> SimpleProcessor:
        return SimpleProcessor(cameras=["front", "wrist"])
