#include <pybind11/pybind11.h>
#include "np2mat/ndarray_converter.h"
#include "cxxlinemod.h"
namespace py = pybind11;

PYBIND11_MODULE(cxxlinemod_pybind, m) {
    NDArrayConverter::init_numpy();
    py::class_<poseRefine>(m, "poseRefine")
        .def(py::init<>())
        .def("getResidual", &poseRefine::getResidual)
        .def("process", &poseRefine::process);
}
