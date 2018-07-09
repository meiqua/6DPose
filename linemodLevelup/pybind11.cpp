#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "np2mat/ndarray_converter.h"
#include "linemodLevelup.h"
namespace py = pybind11;

PYBIND11_MODULE(linemodLevelup_pybind, m) {
    NDArrayConverter::init_numpy();
    py::class_<poseRefine>(m, "poseRefine")
        .def(py::init<>())
        .def("getResidual", &poseRefine::getResidual)
        .def("process", &poseRefine::process)
        .def("getR", &poseRefine::getR)
        .def("getT", &poseRefine::getT);

    py::class_<linemodLevelup::Match>(m,"Match")
            .def(py::init<>())
            .def_readwrite("x",&linemodLevelup::Match::x)
            .def_readwrite("y",&linemodLevelup::Match::y)
            .def_readwrite("similarity",&linemodLevelup::Match::similarity)
            .def_readwrite("class_id",&linemodLevelup::Match::class_id)
            .def_readwrite("template_id",&linemodLevelup::Match::template_id);


    py::class_<linemodLevelup::Detector>(m, "Detector")
        .def(py::init<>())
        .def(py::init<std::vector<int> >())
        .def(py::init<int, std::vector<int> >())
        .def("addTemplate", &linemodLevelup::Detector::addTemplate)
        .def("writeClasses", &linemodLevelup::Detector::writeClasses)
        .def("readClasses", &linemodLevelup::Detector::readClasses)
        .def("match", &linemodLevelup::Detector::match, py::arg("sources"),
             py::arg("threshold"), py::arg("class_ids"), py::arg("masks")=cv::Mat())
        .def("getTemplates", &linemodLevelup::Detector::getTemplates);
}
