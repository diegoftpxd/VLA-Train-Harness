"""Suite CORE contra ``FakeCoreClient`` y contra el ``CoreClient`` real."""

from collections.abc import Iterator

import pytest

from vla_learner.data.core_client import CoreClient
from vla_learner.testing.contracts.core_client_contract import CoreClientContract
from vla_learner.testing.fake_core import FakeCoreClient
from vla_learner.testing.fake_core_server import FakeCoreServer

pytestmark = pytest.mark.fast


class TestFakeCoreClient(CoreClientContract):
    @pytest.fixture
    def client(self) -> FakeCoreClient:
        return FakeCoreClient()


class TestRealCoreClient(CoreClientContract):
    @pytest.fixture
    def client(self) -> Iterator[CoreClient]:
        server = FakeCoreServer()
        address = server.start()
        core_client = CoreClient(address, timeout_s=5.0)
        yield core_client
        core_client.close()
        server.stop()


def test_core04_timeout_sin_servidor():
    client = CoreClient("tcp://127.0.0.1:59998", timeout_s=0.3)
    with pytest.raises(TimeoutError):
        client.reset(["e0"])
    client.close()
