import gc as _gc

# xdsl's IRDL segfaults on GC
_gc.disable()
_gc.set_threshold(0)
_gc.enable = lambda: None
_gc.collect = lambda *a, **kw: 0
