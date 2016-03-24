"""A bounded height priority queue and mapping.

The bounded height priority mapping is a related data structure which
allows for updating the priorities in constant time as well.

"""
from collections.abc import MutableMapping, Sequence
from itertools import chain, islice
from llist import sllist, dllist


class BoundedPriorityQueue(Sequence):
    """Bounded priority queue.

    A bounded height priority queue is a priority queue where the keys
    (priorities) are bounded by a pre-determined value. It consists of an
    array of linked lists. Insertion and finding the maximum can be done in
    constant time, while extracting the maximum is O(C) where C is the
    maximum priority. Its memory requirements are O(C).

    Parameters
    ----------
    bound : int
        The maximum priority that an item in the queue can take.
    items : iterable of tuples, optional
        Items to initialize the queue with. Items are tuples of the form
        ``(priority, value)``.

    Notes
    -----
    The current implementation does not destroy empty linked lists. This is
    more efficient if items with the same priorities often get added and
    removed, but comes at the cost of memory.

    Some operations (like iteration) could be further optimized by a
    constant factor by keeping track of the bottom as well.

    """
    def __init__(self, bound, items=None):
        self._bins = [None] * (bound + 1)
        self._top = 0
        self._len = 0
        if items:
            for item in items:
                self.push(item)

    def __getitem__(self, key):
        return next(islice(self, key, None))

    def __iter__(self):
        return chain.from_iterable(filter(None, self._bins[self._top::-1]))

    def __reversed__(self):
        return chain.from_iterable(filter(None, self._bins[:self._top + 1]))

    def __len__(self):
        return self._len

    def push(self, item):
        self._len += 1
        priority, value = item
        if priority > self._top:
            self._top = priority
        l = self._bins[priority]
        if l is None:
            l = sllist()
            self._bins[priority] = l
        l.append(value)

    def peek(self):
        if not self._len:
            raise IndexError('peek into empty queue')
        return (self._top, self._bins[self._top].first.value)

    def pop(self):
        if not self._len:
            raise IndexError('pop from empty queue')
        self._len -= 1
        top = self._top
        value = self._bins[top].popleft()
        # NOTE Could destroy empty lists here for memory efficiency
        while self._top and not self._bins[self._top]:
            self._top -= 1
        return (top, value)


class BoundedPriorityMapping(MutableMapping):
    """Bounded priority mapping.

    Similar to the bounded heigh priority queue, but uses a secondary data
    structure that allows the priority of keys to be updated. Updating
    happens in constant time, unless the item updated was the maximum
    value, in which case the complexity is O(C).

    Parameters
    ----------
    bound : int
        The maximum priority that an item in the queue can take. Must be
        non-negative.
    items : iterable of tuples, optional
        Items to initialize the queue with. Items are tuples of the form
        ``(priority, value)``.

    Notes
    -----
    It would be possible to internally use a cache that accumulates updates
    to existing keys, and applies all these updates at once when the
    mapping is queried. This could alleviate slowness from repeatedly
    incrementing or decrementing items without querying in between.

    """
    def __init__(self, bound, items=None):
        self._bins = [None] * (bound + 1)
        self._top = 0
        self._len = 0
        self._mapping = {}  # Maps values to (priority, node)
        if items:
            for value, priority in items:
                self[value] = priority

    def __iter__(self):
        return chain.from_iterable(filter(None, self._bins[self._top::-1]))

    def __reversed__(self):
        return chain.from_iterable(filter(None, self._bins[:self._top + 1]))

    def __len__(self):
        return self._len

    def __getitem__(self, value):
        priority, _ = self._mapping[value]
        return priority

    def __setitem__(self, value, priority):
        # If a node already exists, remove it
        if priority < 0:
            raise KeyError('priority cannot be negative')
        if value in self._mapping:
            self._remove_node(value)
        else:
            self._len += 1

        # Get the list, create one if needed
        l = self._bins[priority]
        if l is None:
            l = dllist()
            self._bins[priority] = l
        n = l.append(value)
        self._mapping[value] = (priority, n)

        # Make sure the top pointer is in the right place
        if priority > self._top:
            self._top = priority
        else:
            while not self._bins[self._top]:
                self._top -= 1

    def _remove_node(self, value):
        priority, n = self._mapping[value]
        l = self._bins[priority]
        l.remove(n)

    def __delitem__(self, value):
        self._len -= 1
        self._remove_node(value)
        del self._mapping[value]
        while not self._bins[self._top]:
            self._top -= 1

    def peek(self):
        return (self._top, self._bins[self._top].first.value)

    def pop(self):
        top = self._top
        value = self._bins[top].first.value
        del self[value]
        return (top, value)


class DefaultBoundedPriorityMapping(BoundedPriorityMapping):
    """A bounded priority mapping with a default priority.

    Same as :class:`BoundedPriorityMapping`, but elements not found in the
    mapping have a default priority.

    Parameters
    ----------
    bound : int
        The maximum priority that an item in the queue can take. Must be
        non-negative.
    items : iterable of tuples, optional
        Items to initialize the queue with. Items are tuples of the form
        ``(priority, value)``.
    default_priority : int, optional
        The default priority that missing values take. Must be
        non-negative. Defaults to 0.

    """
    def __init__(self, bound, items=None, default_priority=0):
        super(DefaultBoundedPriorityMapping, self).__init__(bound, items=items)
        self._default_priority = default_priority

    def __getitem__(self, value):
        item = self._mapping.get(value)
        return item[0] if item else self._default_priority
