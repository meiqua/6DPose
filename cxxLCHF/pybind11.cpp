#include <pybind11/pybind11.h>
#include "np2mat/ndarray_converter.h"
#include "lchf.h"
namespace py = pybind11;

PYBIND11_MODULE(cxxLCHF_pybind, m) {
    NDArrayConverter::init_numpy();
}
