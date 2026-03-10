#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    PyObject_HEAD void (*fn)(void);
    PyObject *engine;
    vectorcallfunc vectorcall;
} JitFuncObject;

typedef intptr_t T;

static int arg_to_intptr(PyObject *obj, intptr_t *out, Py_buffer *view, int *is_view) {
    // helper function. convert a python object to intptr_t: buffer objects -> data pointer, ints -> value
    *is_view = 0;

    // try buffer protocol first (numpy arrays, bytearrays, etc.)
    if (PyObject_GetBuffer(obj, view, PyBUF_SIMPLE) == 0) {
        *out = (intptr_t)view->buf;
        *is_view = 1;
        return 0;
    }

    // GetBuffer sets an error on failure, clear it before trying int
    PyErr_Clear();

    if (PyLong_Check(obj)) {
        *out = (intptr_t)PyLong_AsLongLong(obj);
        return ((*out == -1) && PyErr_Occurred()) ? -1 : 0;
    }

    PyErr_Format(PyExc_TypeError, "expected buffer or int, got %.200s", Py_TYPE(obj)->tp_name);
    return -1;
}

static PyObject *JitFunc_vectorcall(PyObject *callable, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    // cpython vectorcall entry point. marshal args, call JIT fn, release buffers

    JitFuncObject *self = (JitFuncObject *)callable;
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if (nargs > 16) {
        PyErr_SetString(PyExc_TypeError, "JitFunc: too many arguments (max 16)");
        return NULL;
    }

    // marshal python args into a flat intptr_t array
    intptr_t a[16];
    Py_buffer views[16];
    Py_ssize_t n_views = 0;
    int ok = 1;
    for (Py_ssize_t i = 0; i < nargs && ok; i++) {
        int is_view;
        ok = (arg_to_intptr(args[i], &a[i], &views[n_views], &is_view) == 0);
        n_views += is_view;
    }

    // always pass 16 args — callee ignores extras per ARM64/x86-64 calling convention
    if (ok) {
        ((void (*)(T, T, T, T, T, T, T, T, T, T, T, T, T, T, T, T))self->fn)(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
    }

    // release any buffer views we acquired
    for (Py_ssize_t i = 0; i < n_views; i++) {
        PyBuffer_Release(&views[i]);
    }
    return ok ? Py_None : NULL;
}

static int JitFunc_init(JitFuncObject *self, PyObject *args, PyObject *kwds) {
    // store fn pointer and hold a ref to the JIT engine to prevent GC
    unsigned long long address;
    PyObject *engine = Py_None;
    static char *kwlist[] = {"address", "engine", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "K|O", kwlist, &address, &engine)) {
        return -1;
    }
    self->fn = (void (*)(void))(uintptr_t)address;
    Py_XDECREF(self->engine);
    Py_INCREF(engine);
    self->engine = engine;
    self->vectorcall = JitFunc_vectorcall;
    return 0;
}

static void JitFunc_dealloc(JitFuncObject *self) {
    Py_XDECREF(self->engine);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyType_Slot JitFunc_slots[] = {
    {Py_tp_init, JitFunc_init},
    {Py_tp_dealloc, JitFunc_dealloc},
    {0, NULL},
};

static PyType_Spec JitFunc_spec = {
    .name = "xnumpy.jitcall.JitFunc",
    .basicsize = sizeof(JitFuncObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    .slots = JitFunc_slots,
};

static PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name = "xnumpy.jitcall",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_jitcall(void) {
    // register JitFunc type with vectorcall support
    PyObject *mod = PyModule_Create(&module_def);
    if (!mod) {
        return NULL;
    }

    PyObject *type = PyType_FromSpec(&JitFunc_spec);
    if (!type) {
        Py_DECREF(mod);
        return NULL;
    }

    // tell cpython where our vectorcall function pointer lives in the struct
    ((PyTypeObject *)type)->tp_vectorcall_offset = offsetof(JitFuncObject, vectorcall);

    if (PyModule_AddObject(mod, "JitFunc", type) < 0) {
        Py_DECREF(type);
        Py_DECREF(mod);
        return NULL;
    }
    return mod;
}
