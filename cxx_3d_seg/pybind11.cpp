#include <pybind11/pybind11.h>
#include "np2mat/ndarray_converter.h"
#include "cxx_3d_seg.h"
namespace py = pybind11;

PYBIND11_MODULE(cxx_3d_seg_pybind, m) {
    NDArrayConverter::init_numpy();
}
