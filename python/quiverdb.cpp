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
        .def("get_vector", [](const HNSWIndex& self, uint64_t id) {
                std::vector<float> vec = self.get_vector(id);
                return py::array_t<float>(
                    {vec.size()}, // Shape
                    {sizeof(float)}, // Strides
                    vec.data() // Data pointer (copies data)
                );
            },
            py::arg("id"),
            "Retrieves the vector associated with the given ID as a numpy array")
        .def("search", [](const HNSWIndex& self, py::array_t<float> query_array, size_t k) {
                py::buffer_info buf = query_array.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("Query vector must be a 1-dimensional array");
                }
                if (buf.size != self.dimension()) {
                    throw std::runtime_error("Query dimension mismatch");
                }

                std::vector<HNSWSearchResult> results = self.search(static_cast<const float*>(buf.ptr), k);
                
                // Create numpy arrays for IDs and distances
                py::array_t<uint64_t> ids(static_cast<ssize_t>(results.size()));
                py::array_t<float> dists(static_cast<ssize_t>(results.size()));

                auto ids_ptr = ids.mutable_unchecked<1>();
                auto dists_ptr = dists.mutable_unchecked<1>();

                for (size_t i = 0; i < results.size(); ++i) {
                    ids_ptr(i) = results[i].id;
                    dists_ptr(i) = results[i].distance;
                }

                return py::make_tuple(ids, dists);
            },
            py::arg("query_vector"), py::arg("k"),
            "Searches for k nearest neighbors. Returns a tuple (ids, distances).")
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