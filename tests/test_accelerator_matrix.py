import os

import pytest

from audio_super_resolution import available_devices, resolve_device

RUN_ACCELERATOR_MATRIX = os.environ.get("AUDIO_SUPER_RESOLUTION_RUN_ACCELERATOR_MATRIX") == "1"


@pytest.mark.skipif(not RUN_ACCELERATOR_MATRIX, reason="set AUDIO_SUPER_RESOLUTION_RUN_ACCELERATOR_MATRIX=1")
def test_requested_accelerator_device_is_available() -> None:
    requested = os.environ.get("AUDIO_SUPER_RESOLUTION_ACCELERATOR_DEVICE", "auto")

    resolved = resolve_device(requested)
    devices = {device.name: device for device in available_devices()}

    assert resolved
    if requested != "auto":
        assert devices[requested].available is True
