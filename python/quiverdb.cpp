#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h> // For std::vector<std::vector<size_t>> - not strictly needed for this type of project, but was present. Remove if unnecessary.
#include <pybind11/operators.h>
#include <pybind11/numpy.h> // Added for NumPy array support

#include "core/hnsw_index.h"

namespace py = pybind11;
using namespace quiverdb;

// Define a Python-friendly representation of HNSWSearchResult
struct PyHNSWSearchResult {
    uint64_t id;
    float distance;

    // For printing in Python
    std::string toString() const {
        return "HNSWSearchResult(id=" + std::to_string(id) + ", distance=" + std::to_string(distance) + ")";
    }
};

PYBIND11_MODULE(quiverdb_py, m) {
    m.doc() = "pybind11 wrapper for QuiverDB HNSWIndex"; // Optional module docstring

    // Bind HNSWDistanceMetric enum
    py::enum_<HNSWDistanceMetric>(m, "HNSWDistanceMetric")
        .value("L2", HNSWDistanceMetric::L2)
        .value("COSINE", HNSWDistanceMetric::COSINE)
        .value("DOT", HNSWDistanceMetric::DOT)
        .export_values(); // Exports enum values to the module scope

    // Bind HNSWSearchResult struct
    py::class_<PyHNSWSearchResult>(m, "HNSWSearchResult")
        .def_readwrite("id", &PyHNSWSearchResult::id)
        .def_readwrite("distance", &PyHNSWSearchResult::distance)
        .def("__repr__", &PyHNSWSearchResult::toString);

    // Bind HNSWIndex class
    py::class_<HNSWIndex>(m, "HNSWIndex")
        .def(py::init<size_t, HNSWDistanceMetric, size_t, size_t, size_t, uint32_t>(),
             py::arg("dimension"),
             py::arg("metric") = HNSWDistanceMetric::L2,
             py::arg("max_elements") = 100000,
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("random_seed") = 42)
        .def("add", [](HNSWIndex& self, uint64_t id, py::array_t<float> vector_array) {
                py::buffer_info buf = vector_array.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("Vector must be a 1-dimensional array");
                }
                if (buf.size != self.dimension()) {
                    throw std::runtime_error("Vector dimension mismatch");
                }
                self.add(id, static_cast<const float*>(buf.ptr));
            },
            py::arg("id"), py::arg("vector"),
            "Adds a vector to the index")
        .def("search", [](const HNSWIndex& self, const std::vector<float>& query_vector, size_t k) {
                // Convert C++ HNSWSearchResult to Python-friendly PyHNSWSearchResult
                std::vector<HNSWSearchResult> cpp_results = self.search(query_vector.data(), k);
                std::vector<PyHNSWSearchResult> py_results;
                py_results.reserve(cpp_results.size());
                for (const auto& res : cpp_results) {
                    py_results.push_back({res.id, res.distance});
                }
                return py_results;
            },
            py::arg("query_vector"), py::arg("k"),
            "Searches for k nearest neighbors")
        .def("size", &HNSWIndex::size, "Returns the number of vectors in the index")
        .def("dimension", &HNSWIndex::dimension, "Returns the dimension of stored vectors")
        .def("capacity", &HNSWIndex::capacity, "Returns the maximum capacity of the index")
        .def("contains", &HNSWIndex::contains, py::arg("id"), "Checks if a vector with given ID exists")
        .def("set_ef_search", &HNSWIndex::set_ef_search, py::arg("ef"), "Sets the ef parameter for search")
        .def("get_ef_search", &HNSWIndex::get_ef_search, "Returns the current ef_search parameter")
        .def("save", &HNSWIndex::save, py::arg("filename"), "Saves the index to a binary file")
        .def_static("load", (std::unique_ptr<HNSWIndex> (*)(const std::string&)) &HNSWIndex::load,
                    py::arg("filename"), "Loads the index from a binary file",
                    py::return_value_policy::take_ownership);
}