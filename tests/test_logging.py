import genoray._core as core


def test_event_channel_roundtrip():
    rx = core.PyEventReceiver(flush_every=1)
    # No producer yet: recv_timeout returns None promptly.
    assert rx.recv_timeout(1) is None
