#include <pybind11/pybind11.h>
#include "np2mat/ndarray_converter.h"
#include "cxx_3d_seg.h"
namespace py = pybind11;

PYBIND11_MODULE(cxx_3d_seg_pybind, m) {
    NDArrayConverter::init_numpy();
    m.def("convex_cloud_seg", &convex_cloud_seg, "all in float32");
    m.def("depth2cloud", &depth2cloud, "all in float32");
    m.def("pose_estimation", &pose_estimation, "all in float32",
          py::arg("cloud") = cv::Mat(),
          py::arg("ply_model") = "",
          py::arg("LCP_thresh") = 0.2,
          py::arg("ICP_thresh") = 100,
          py::arg("use_pcs") = true,
          py::arg("pcs_seconds") = 1,
          py::arg("use_icp") = true,
          py::arg("cloud_icp_size") = 1000,
          py::arg("model_icp_size") = 10000);
}
