#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core/hnsw_index.h"
#include "core/vector_store.h"
#include "core/mmap_vector_store.h"
#include "core/version.h"

namespace py = pybind11;
using namespace quiverdb;

PYBIND11_MODULE(quiverdb_py, m) {
    m.doc() = "QuiverDB - Embeddable vector database for edge AI";

    // Version info
    m.attr("__version__") = VERSION_STRING;
    m.attr("VERSION_MAJOR") = VERSION_MAJOR;
    m.attr("VERSION_MINOR") = VERSION_MINOR;
    m.attr("VERSION_PATCH") = VERSION_PATCH;

    // Bind HNSWDistanceMetric enum
    py::enum_<HNSWDistanceMetric>(m, "HNSWDistanceMetric")
        .value("L2", HNSWDistanceMetric::L2)
        .value("COSINE", HNSWDistanceMetric::COSINE)
        .value("DOT", HNSWDistanceMetric::DOT)
        .export_values(); // Exports enum values to the module scope

    // Bind HNSWIndex class
    py::class_<HNSWIndex>(m, "HNSWIndex")
        .def(py::init<size_t, HNSWDistanceMetric, size_t, size_t, size_t, uint32_t>(),
             py::arg("dimension"),
             py::arg("metric") = HNSWDistanceMetric::L2,
             py::arg("max_elements") = 100000,
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("random_seed") = 42)
        .def("add", [](HNSWIndex& self, uint64_t id, py::array_t<float, py::array::c_style | py::array::forcecast> vector_array) {
                py::buffer_info buf = vector_array.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("Vector must be a 1-dimensional array");
                }
                if (static_cast<size_t>(buf.size) != self.dimension()) {
                    throw std::runtime_error("Vector dimension mismatch");
                }
                // Release GIL during potentially long C++ operation
                py::gil_scoped_release release;
                self.add(id, static_cast<const float*>(buf.ptr));
            },
            py::arg("id"), py::arg("vector"),
            "Adds a vector to the index")
        .def("get_vector", [](const HNSWIndex& self, uint64_t id) {
                std::vector<float> vec = self.get_vector(id);
                // Create array that owns its data by using a capsule to prevent use-after-free
                auto* vec_ptr = new std::vector<float>(std::move(vec));
                auto capsule = py::capsule(vec_ptr, [](void* p) {
                    delete static_cast<std::vector<float>*>(p);
                });
                return py::array_t<float>(
                    {vec_ptr->size()},
                    {sizeof(float)},
                    vec_ptr->data(),
                    capsule  // Prevents deallocation until array is destroyed
                );
            },
            py::arg("id"),
            "Retrieves the vector associated with the given ID as a numpy array")
        .def("search", [](const HNSWIndex& self, py::array_t<float, py::array::c_style | py::array::forcecast> query_array, size_t k) {
                py::buffer_info buf = query_array.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("Query vector must be a 1-dimensional array");
                }
                if (static_cast<size_t>(buf.size) != self.dimension()) {
                    throw std::runtime_error("Query dimension mismatch");
                }

                std::vector<HNSWSearchResult> results;
                {
                    // Release GIL during search
                    py::gil_scoped_release release;
                    results = self.search(static_cast<const float*>(buf.ptr), k);
                }

                // Create numpy arrays for IDs and distances (requires GIL)
                py::array_t<uint64_t> ids(static_cast<py::ssize_t>(results.size()));
                py::array_t<float> dists(static_cast<py::ssize_t>(results.size()));

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
        .def("save", [](const HNSWIndex& self, const std::string& filename) {
                py::gil_scoped_release release;
                self.save(filename);
            },
            py::arg("filename"), "Saves the index to a binary file")
        .def_static("load", [](const std::string& filename) {
                py::gil_scoped_release release;
                return HNSWIndex::load(filename);
            },
            py::arg("filename"), "Loads the index from a binary file",
            py::return_value_policy::take_ownership);

    // Bind DistanceMetric enum (for VectorStore and MMapVectorStore)
    py::enum_<DistanceMetric>(m, "DistanceMetric")
        .value("L2", DistanceMetric::L2)
        .value("COSINE", DistanceMetric::COSINE)
        .value("DOT", DistanceMetric::DOT)
        .export_values();

    // Bind VectorStore class (brute-force, thread-safe)
    py::class_<VectorStore>(m, "VectorStore")
        .def(py::init<size_t, DistanceMetric>(),
             py::arg("dimension"),
             py::arg("metric") = DistanceMetric::L2,
             "Creates a new in-memory vector store")
        .def("add", [](VectorStore& self, uint64_t id, py::array_t<float, py::array::c_style | py::array::forcecast> vector_array) {
                py::buffer_info buf = vector_array.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("Vector must be a 1-dimensional array");
                }
                if (static_cast<size_t>(buf.size) != self.dimension()) {
                    throw std::runtime_error("Vector dimension mismatch");
                }
                py::gil_scoped_release release;
                self.add(id, static_cast<const float*>(buf.ptr));
            },
            py::arg("id"), py::arg("vector"),
            "Adds a vector to the store")
        .def("get", [](const VectorStore& self, uint64_t id) -> py::object {
                // Use get_copy() for thread-safe copy while holding the lock
                std::vector<float> vec = self.get_copy(id);
                if (vec.empty()) {
                    return py::none();
                }
                // Move the copy into heap-allocated storage for the capsule
                auto* vec_copy = new std::vector<float>(std::move(vec));
                auto capsule = py::capsule(vec_copy, [](void* p) {
                    delete static_cast<std::vector<float>*>(p);
                });
                return py::array_t<float>(
                    {vec_copy->size()},
                    {sizeof(float)},
                    vec_copy->data(),
                    capsule  // Prevents deallocation until array is destroyed
                );
            },
            py::arg("id"),
            "Gets a vector by ID, returns None if not found")
        .def("search", [](const VectorStore& self, py::array_t<float, py::array::c_style | py::array::forcecast> query_array, size_t k) {
                py::buffer_info buf = query_array.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("Query vector must be a 1-dimensional array");
                }
                if (static_cast<size_t>(buf.size) != self.dimension()) {
                    throw std::runtime_error("Query dimension mismatch");
                }

                std::vector<SearchResult> results;
                {
                    py::gil_scoped_release release;
                    results = self.search(static_cast<const float*>(buf.ptr), k);
                }

                py::array_t<uint64_t> ids(static_cast<py::ssize_t>(results.size()));
                py::array_t<float> dists(static_cast<py::ssize_t>(results.size()));
                auto ids_ptr = ids.mutable_unchecked<1>();
                auto dists_ptr = dists.mutable_unchecked<1>();

                for (size_t i = 0; i < results.size(); ++i) {
                    ids_ptr(i) = results[i].id;
                    dists_ptr(i) = results[i].distance;
                }

                return py::make_tuple(ids, dists);
            },
            py::arg("query_vector"), py::arg("k"),
            "Searches for k nearest neighbors. Returns (ids, distances).")
        .def("remove", &VectorStore::remove, py::arg("id"), "Removes a vector by ID")
        .def("update", [](VectorStore& self, uint64_t id, py::array_t<float, py::array::c_style | py::array::forcecast> vector_array) {
                py::buffer_info buf = vector_array.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("Vector must be a 1-dimensional array");
                }
                if (static_cast<size_t>(buf.size) != self.dimension()) {
                    throw std::runtime_error("Vector dimension mismatch");
                }
                py::gil_scoped_release release;
                return self.update(id, static_cast<const float*>(buf.ptr));
            },
            py::arg("id"), py::arg("vector"),
            "Updates an existing vector")
        .def("size", &VectorStore::size, "Returns the number of vectors")
        .def("dimension", &VectorStore::dimension, "Returns the dimension")
        .def("contains", &VectorStore::contains, py::arg("id"), "Checks if ID exists")
        .def("clear", &VectorStore::clear, "Removes all vectors")
        .def("reserve", &VectorStore::reserve, py::arg("capacity"), "Pre-allocates space");

    // Bind MMapVectorStoreBuilder class
    py::class_<MMapVectorStoreBuilder>(m, "MMapVectorStoreBuilder")
        .def(py::init<size_t, DistanceMetric>(),
             py::arg("dimension"),
             py::arg("metric") = DistanceMetric::L2,
             "Creates a new builder for memory-mapped vector store")
        .def("add", [](MMapVectorStoreBuilder& self, uint64_t id, py::array_t<float, py::array::c_style | py::array::forcecast> vector_array) {
                py::buffer_info buf = vector_array.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("Vector must be a 1-dimensional array");
                }
                if (static_cast<size_t>(buf.size) != self.dimension()) {
                    throw std::runtime_error("Vector dimension mismatch");
                }
                self.add(id, static_cast<const float*>(buf.ptr));
            },
            py::arg("id"), py::arg("vector"),
            "Adds a vector to the builder")
        .def("save", [](const MMapVectorStoreBuilder& self, const std::string& filename) {
                py::gil_scoped_release release;
                self.save(filename);
            },
            py::arg("filename"),
            "Saves to a memory-mappable file")
        .def("size", &MMapVectorStoreBuilder::size, "Returns the number of vectors")
        .def("dimension", &MMapVectorStoreBuilder::dimension, "Returns the dimension")
        .def("reserve", &MMapVectorStoreBuilder::reserve, py::arg("capacity"), "Pre-allocates space");

    // Bind MMapVectorStore class (read-only, memory-mapped)
    py::class_<MMapVectorStore>(m, "MMapVectorStore")
        .def(py::init<const std::string&>(),
             py::arg("filename"),
             "Opens a memory-mapped vector store file")
        .def("get", [](const MMapVectorStore& self, uint64_t id) -> py::object {
                const float* ptr = self.get(id);
                if (ptr == nullptr) {
                    return py::none();
                }
                // Return a view into the mapped memory (zero-copy!)
                return py::array_t<float>(
                    {self.dimension()},
                    {sizeof(float)},
                    ptr,
                    py::cast(&self)  // Keep store alive while array exists
                );
            },
            py::arg("id"),
            "Gets a vector by ID (zero-copy from mmap), returns None if not found")
        .def("search", [](const MMapVectorStore& self, py::array_t<float, py::array::c_style | py::array::forcecast> query_array, size_t k) {
                py::buffer_info buf = query_array.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("Query vector must be a 1-dimensional array");
                }
                if (static_cast<size_t>(buf.size) != self.dimension()) {
                    throw std::runtime_error("Query dimension mismatch");
                }

                std::vector<SearchResult> results;
                {
                    py::gil_scoped_release release;
                    results = self.search(static_cast<const float*>(buf.ptr), k);
                }

                py::array_t<uint64_t> ids(static_cast<py::ssize_t>(results.size()));
                py::array_t<float> dists(static_cast<py::ssize_t>(results.size()));
                auto ids_ptr = ids.mutable_unchecked<1>();
                auto dists_ptr = dists.mutable_unchecked<1>();

                for (size_t i = 0; i < results.size(); ++i) {
                    ids_ptr(i) = results[i].id;
                    dists_ptr(i) = results[i].distance;
                }

                return py::make_tuple(ids, dists);
            },
            py::arg("query_vector"), py::arg("k"),
            "Searches for k nearest neighbors. Returns (ids, distances).")
        .def("size", &MMapVectorStore::size, "Returns the number of vectors")
        .def("dimension", &MMapVectorStore::dimension, "Returns the dimension")
        .def("contains", &MMapVectorStore::contains, py::arg("id"), "Checks if ID exists");
}