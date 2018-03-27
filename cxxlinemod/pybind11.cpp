#include <pybind11/pybind11.h>
#include "cxxlinemod.h"
#include "ndarray_converter.h"
namespace py = pybind11;

PYBIND11_PLUGIN(cxxlinemod)
{
    NDArrayConverter::init_numpy();

    py::module m("cxxlinemod", "pybind11 cxxlinemod");
    m.def("read_image", &read_image, "A function that read an image",
        py::arg("image"));

    m.def("show_image", &show_image, "A function that show an image",
        py::arg("image"));

    m.def("passthru", &passthru, "Passthru function", py::arg("image"));
    m.def("clone", &cloneimg, "Clone function", py::arg("image"));

    return m.ptr();
}
