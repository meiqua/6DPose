#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "np2mat/ndarray_converter.h"
#include "linemodLevelup.h"
namespace py = pybind11;

PYBIND11_MODULE(linemodLevelup_pybind, m) {
    NDArrayConverter::init_numpy();

    py::class_<poseRefine>(m, "poseRefine")
        .def(py::init<>())
        .def_readwrite("result_refined",&poseRefine::result_refined)
        .def_readwrite("inlier_rmse",&poseRefine::inlier_rmse)
        .def_readwrite("fitness",&poseRefine::fitness)
        .def("get_depth_edge", &poseRefine::get_depth_edge)
        .def("process", &poseRefine::process);

    py::class_<linemodLevelup::Match>(m,"Match")
            .def(py::init<>())
            .def_readwrite("x",&linemodLevelup::Match::x)
            .def_readwrite("y",&linemodLevelup::Match::y)
            .def_readwrite("similarity",&linemodLevelup::Match::similarity)
            .def_readwrite("class_id",&linemodLevelup::Match::class_id)
            .def_readwrite("template_id",&linemodLevelup::Match::template_id);

    py::class_<linemodLevelup::Template>(m,"Template")
            .def(py::init<>())
            .def_readwrite("width",&linemodLevelup::Template::width)
            .def_readwrite("height",&linemodLevelup::Template::height)
            .def_readwrite("pyramid_level",&linemodLevelup::Template::pyramid_level);


    py::class_<linemodLevelup::Detector>(m, "Detector")
        .def(py::init<>())
        .def(py::init<std::vector<int>, int>())
        .def(py::init<int, std::vector<int>, int>())
        .def("addTemplate", &linemodLevelup::Detector::addTemplate)
        .def("writeClasses", &linemodLevelup::Detector::writeClasses)
        .def("clear_classes", &linemodLevelup::Detector::clear_classes)
        .def("readClasses", &linemodLevelup::Detector::readClasses)
        .def("match", &linemodLevelup::Detector::match, py::arg("sources"),
             py::arg("threshold"), py::arg("active_ratio"), py::arg("class_ids"),
             py::arg("dep_anchors"), py::arg("dep_range"), py::arg("masks")=cv::Mat())
        .def("getTemplates", &linemodLevelup::Detector::getTemplates)
            .def("numTemplates", &linemodLevelup::Detector::numTemplates);
}
