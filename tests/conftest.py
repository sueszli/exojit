import gc as _gc

# xDSL IRDL holds raw ctypes pointers. GC finalizer ordering -> dangling ptr -> segfault
_gc.disable()
_gc.set_threshold(0)
_gc.enable = lambda: None
_gc.collect = lambda *a, **kw: 0
