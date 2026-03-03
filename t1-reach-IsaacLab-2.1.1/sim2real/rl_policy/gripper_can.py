import time
import can

interface = 'socketcan'
channel = 'vcan0'


def producer(id):
    """:param id: Spam the bus with messages including the data id."""
    bus = can.Bus(channel=channel, interface=interface)
    for i in range(10):
        msg = can.Message(arbitration_id=0xc0ffee, data=[id, i, 0, 1, 3, 1, 4, 1], is_extended_id=True)
        bus.send(msg)

    time.sleep(1)
    bus.shutdown()


producer(10)
