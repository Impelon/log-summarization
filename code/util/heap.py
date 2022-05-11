import heapq
import collections.abc
from collections import namedtuple

HeapItem = namedtuple("HeapItem", ["key", "insertion_rank", "value"])

class Heap(collections.abc.Iterable, collections.abc.Sized):

    def __init__(self, iterable=None, key=None):
        """
        Initialization of a Heap.

        Args:
            iterable: An optional iterable to initialize from.
                If this is a falsy value (e.g. None), an empty heap will be created.
            key: An optional custom key function to specify the heap order.
        """
        if not key:
            def key(x): return x
        self.key = key
        self.heap_items = []
        if iterable:
            self.heap_items = [HeapItem(self.key(item), i, item) for i, item in enumerate(iterable)]
        self.insertion_count = len(self.heap_items)
        heapq.heapify(self.heap_items)

    def __repr__(self):
        return "{}({})".format(type(self).__name__, repr(list(self)))

    def push(self, item):
        self.insertion_count += 1
        heapq.heappush(self.heap_items, HeapItem(self.key(item), self.insertion_count, item))

    def pop(self):
        return heapq.heappop(self.heap_items).value

    def pushpop(self, item):
        self.insertion_count += 1
        return heapq.heappushpop(self.heap_items, HeapItem(self.key(item), self.insertion_count, item)).value

    def poppush(self, item):
        self.insertion_count += 1
        return heapq.heapreplace(self.heap_items, HeapItem(self.key(item), self.insertion_count, item)).value

    def __len__(self):
        return len(self.heap_items)

    def __iter__(self):
        return (x.value for x in self.heap_items)

    def consume(self):
        while self:
            yield self.pop()
