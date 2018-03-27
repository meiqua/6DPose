#include <pybind11/pybind11.h>
#include "np2mat/ndarray_converter.h"
#include "cxxlinemod.h"
namespace py = pybind11;

PYBIND11_MODULE(cxxlinemod_pybind, m) {
    NDArrayConverter::init_numpy();
    m.def("read_image", &read_image, "A function that read an image",
        py::arg("image"));

    m.def("show_image", &show_image, "A function that show an image",
        py::arg("image"));

    m.def("passthru", &passthru, "Passthru function", py::arg("image"));
    m.def("clone", &cloneimg, "Clone function", py::arg("image"));
}
