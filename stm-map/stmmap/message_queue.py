import heapq as hq
import itertools

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(order=True)
class MessagePriority:
    msg_id: Tuple[int, ...] = field(compare=False)
    priority: float


class MessageQueue:
    """A priority queue ensures unique elements"""

    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = "<removed-task>"  # placeholder for a removed task
        self.unique_counter = 0

    def put(self, msg_priority: MessagePriority):
        """Add a new message or update the priority of an existing message."""
        if msg_priority.msg_id in self.entry_finder:
            msg_priority.priority += self._remove_task(msg_priority.msg_id)
        else:
            self.unique_counter += 1
        self.entry_finder[msg_priority.msg_id] = msg_priority
        hq.heappush(self.pq, msg_priority)

    def _remove_task(self, msg_id: Tuple[int, ...]):
        """Mark an existing task as REMOVED.  Raise KeyError if not found."""
        msg_priority = self.entry_finder.pop(msg_id)
        msg_priority.msg_id = self.REMOVED
        return msg_priority.priority

    def get(self):
        """Remove and return the lowest priority task. Raise KeyError if empty."""
        while self.pq:
            msg_priority = hq.heappop(self.pq)
            if msg_priority.msg_id is not self.REMOVED:
                del self.entry_finder[msg_priority.msg_id]
                self.unique_counter -= 1
                return msg_priority
        raise KeyError("pop from an empty priority queue")

    def empty(self):
        return self.unique_counter == 0


if __name__ == "__main__":
    # Just for sanity checking
    m1 = MessagePriority((0, 0, 1), -1)
    m2 = MessagePriority((0, 1, 1), -5)
    m3 = MessagePriority((0, 0, 1), -6)
    mq = MessageQueue()
    mq.put(m1)
    mq.put(m2)
    mq.put(m3)

    m = mq.get()
    assert m.msg_id == (0, 0, 1) and m.priority == -7
