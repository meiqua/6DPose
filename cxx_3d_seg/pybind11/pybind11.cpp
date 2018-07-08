#include <pybind11/pybind11.h>
#include "np2mat/ndarray_converter.h"
#include "cxx_3d_seg.h"
namespace py = pybind11;

PYBIND11_MODULE(cxx_3d_seg_pybind, m) {
    NDArrayConverter::init_numpy();

    py::class_<cxx_3d_seg::convex_result>(m, "convex_result")
        .def(py::init<>())
        .def("getIndices", &cxx_3d_seg::convex_result::getIndices)
        .def("getCloud", &cxx_3d_seg::convex_result::getCloud)
        .def("getNormal", &cxx_3d_seg::convex_result::getNormal);

    m.def("convex_cloud_seg", &cxx_3d_seg::convex_cloud_seg, "all in float32");
    m.def("pose_estimation", &cxx_3d_seg::pose_estimation, "all in float32",
          py::arg("cloud") = cv::Mat(),
          py::arg("ply_model") = "",
          py::arg("pcs_seconds") = 1,
          py::arg("LCP_thresh") = 0.5);
}
