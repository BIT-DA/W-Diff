from queue import Queue


class MyQueue:
    def __init__(self, maxsize):
        super(MyQueue, self).__init__()
        self.pool = Queue(maxsize)

    def put_item(self, item):
        if self.pool.full():
            self.pool.get()
        self.pool.put(item)

    def get_item_by_idx(self, idx):
        assert idx >=0 and idx < self.pool.qsize()
        return self.pool.queue[idx]

    def get_all_items(self):
        return list(self.pool.queue)

    def len(self):
        return self.pool.qsize()

    def init(self):
        while not self.pool.empty():
            self.pool.get()

