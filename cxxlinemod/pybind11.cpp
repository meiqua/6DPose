#include <pybind11/pybind11.h>
#include "np2mat/ndarray_converter.h"
#include "cxxlinemod.h"
namespace py = pybind11;

PYBIND11_MODULE(cxxlinemod_pybind, m) {
    NDArrayConverter::init_numpy();
    m.def("depth2pc", &depth2pc, "depth to point cloud",
        py::arg("depth"), py::arg("K"));
}
