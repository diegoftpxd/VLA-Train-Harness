"""Suite ENV contra el ``FakeEnvironment`` y contra el ``EnvironmentClient``."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from vla_core.environments import Environment, EnvironmentClient
from vla_core.testing.contracts import EnvironmentContract
from vla_core.testing.fakes import FakeEnvironment, serve_environment

pytestmark = pytest.mark.fast


class TestFakeEnvironment(EnvironmentContract):
    @pytest.fixture
    def environment(self) -> Environment:
        return FakeEnvironment()


class TestEnvironmentClient(EnvironmentContract):
    @pytest.fixture
    def environment(self) -> Iterator[Environment]:
        server = serve_environment(FakeEnvironment())
        address = server.start()
        client = EnvironmentClient(address, timeout_s=5.0)
        try:
            yield client
        finally:
            client.close()
            server.stop()
