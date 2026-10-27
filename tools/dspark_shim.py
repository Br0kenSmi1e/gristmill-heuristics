"""Serial shims for dummy_spark on py3.12."""
import collections, collections.abc
for _n in ("Iterable","Mapping","MutableMapping","Sequence","Callable","Hashable","Set","MutableSequence","MutableSet","ItemsView","KeysView","ValuesView"):
    if not hasattr(collections,_n): setattr(collections,_n,getattr(collections.abc,_n))
import dummy_spark.rdd as _rdd

def aggregate(self, zero, seq_op, comb_op):
    acc = zero
    for x in self.collect():
        acc = seq_op(acc, x)
    return acc
_rdd.RDD.aggregate = aggregate

def treeAggregate(self, zero, seq_op, comb_op, depth=2):
    return aggregate(self, zero, seq_op, comb_op)
_rdd.RDD.treeAggregate = treeAggregate

class _Bcast:
    __slots__ = ("value",)
    def __init__(self, v): self.value = v
    def unpersist(self, *a, **k): pass

import dummy_spark.context as _ctx
_ctx.SparkContext.broadcast = lambda self, v: _Bcast(v)

import functools, itertools

def _collect(self):
    return list(self.collect())

_rdd.RDD.repartition = lambda self, n=None: self
_rdd.RDD.coalesce = lambda self, n=None, shuffle=False: self
_rdd.RDD.repartitionAndSortWithinPartitions = lambda self, *a, **k: self
_rdd.RDD.reduce = lambda self, f: functools.reduce(f, _collect(self))
_rdd.RDD.treeReduce = lambda self, f, depth=2: functools.reduce(f, _collect(self))
_rdd.RDD.fold = lambda self, z, op: functools.reduce(op, _collect(self), z)
_rdd.RDD.collectAsMap = lambda self: dict(_collect(self))
_rdd.RDD.keys = lambda self: self.map(lambda kv: kv[0])
_rdd.RDD.values = lambda self: self.map(lambda kv: kv[1])
_rdd.RDD.countByKey = lambda self: collections.Counter(k for k, _ in _collect(self))
_rdd.RDD.countByValue = lambda self: collections.Counter(_collect(self))
def _reduceByKey(self, func):
    d = {}
    for k, v in _collect(self):
        d[k] = func(d[k], v) if k in d else v
    return self.ctx.parallelize(list(d.items()))
_rdd.RDD.reduceByKey = _reduceByKey
def _aggregateByKey(self, zero, seq, comb, numPartitions=None):
    d = {}
    for k, v in _collect(self):
        d[k] = seq(d[k], v) if k in d else seq(zero, v)
    return self.ctx.parallelize(list(d.items()))
_rdd.RDD.aggregateByKey = _aggregateByKey
_rdd.RDD.foldByKey = lambda self, z, f, numPartitions=None: _aggregateByKey(self, z, f, f)
def _combineByKey(self, create, merge_val, merge_comb, numPartitions=None):
    d = {}
    for k, v in _collect(self):
        d[k] = merge_val(d[k], v) if k in d else create(v)
    return self.ctx.parallelize(list(d.items()))
_rdd.RDD.combineByKey = _combineByKey
_rdd.RDD.groupByKey = lambda self, *a, **k: _combineByKey(self, lambda v: [v], lambda acc, v: acc + [v], lambda a, b: a + b)
_rdd.RDD.mapValues = lambda self, f: self.map(lambda kv: (kv[0], f(kv[1])))
_rdd.RDD.subtract = lambda self, other, numPartitions=None: self.filter(lambda x: x not in set(other.collect()))
def _join(self, other, numPartitions=None):
    right = dict(other.collect())
    return self.ctx.parallelize([(k, (v, right[k])) for k, v in _collect(self) if k in right])
_rdd.RDD.join = _join
_rdd.RDD.keyBy = lambda self, f: self.map(lambda x: (f(x), x))
_rdd.RDD.max = lambda self, key=None: max(_collect(self), key=key)
_rdd.RDD.min = lambda self, key=None: min(_collect(self), key=key)
_rdd.RDD.top = lambda self, n, key=None: sorted(_collect(self), key=key, reverse=True)[:n]
_rdd.RDD.takeOrdered = lambda self, n, key=None: sorted(_collect(self), key=key)[:n]
