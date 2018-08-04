#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "np2mat/ndarray_converter.h"
#include "forest.h"
namespace py = pybind11;

PYBIND11_MODULE(cxxLCHF_pybind, m) {
    NDArrayConverter::init_numpy();

    // class
    py::class_<Linemod_feature>(m, "Linemod_feature")
        .def(py::init<cv::Mat, cv::Mat>())
        .def(py::init<cv::Mat, cv::Mat, cv::Mat>())
        .def("constructEmbedding", &Linemod_feature::constructEmbedding)
        .def("constructResponse", &Linemod_feature::constructResponse);
    py::class_<Info>(m, "Info")
            .def(py::init<>())
            .def_readwrite("id", &Info::id)
            .def_readwrite("rpy", &Info::rpy)
            .def_readwrite("t", &Info::t);
    py::class_<Forest<Linemod_feature>>(m, "linemod_forest");

    // func
    m.def("lchf_model_train", &lchf_model::train);
    m.def("lchf_model_predict", &lchf_model::predict);

    m.def("getLeaf_feats_map", &lchf_model::getLeaf_feats_map);

    m.def("get_feats_from_scene", &lchf_model::get_feats_from_scene);

    m.def("lchf_model_saveForest", &lchf_model::saveForest);
    m.def("lchf_model_loadForest", &lchf_model::loadForest);

    m.def("lchf_model_saveFeatures", &lchf_model::saveFeatures);
    m.def("lchf_model_loadFeatures", &lchf_model::loadFeatures);

    m.def("lchf_model_saveInfos", &lchf_model::saveInfos);
    m.def("lchf_model_loadInfos", &lchf_model::loadInfos);
}
