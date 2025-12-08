"""
Python binding tests for QuiverDB HNSWIndex.

Run with: uv run pytest tests/test_python_bindings.py -v
"""

import pytest
import numpy as np
import tempfile
import os


def test_import():
    """Test that the module can be imported."""
    import quiverdb_py
    assert hasattr(quiverdb_py, 'HNSWIndex')
    assert hasattr(quiverdb_py, 'HNSWDistanceMetric')
    assert hasattr(quiverdb_py, 'HNSWSearchResult')


def test_version():
    """Test that version info is accessible."""
    import quiverdb_py
    assert hasattr(quiverdb_py, '__version__')
    assert isinstance(quiverdb_py.__version__, str)
    assert quiverdb_py.__version__ == "0.1.0"
    assert quiverdb_py.VERSION_MAJOR == 0
    assert quiverdb_py.VERSION_MINOR == 1
    assert quiverdb_py.VERSION_PATCH == 0


def test_distance_metrics():
    """Test that distance metric enum values are accessible."""
    import quiverdb_py
    assert quiverdb_py.HNSWDistanceMetric.L2 is not None
    assert quiverdb_py.HNSWDistanceMetric.COSINE is not None
    assert quiverdb_py.HNSWDistanceMetric.DOT is not None


def test_create_index_default():
    """Test creating an index with default parameters."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(dimension=128)
    assert index.size() == 0
    assert index.dimension() == 128
    assert index.capacity() == 100000


def test_create_index_custom():
    """Test creating an index with custom parameters."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(
        dimension=64,
        metric=quiverdb_py.HNSWDistanceMetric.COSINE,
        max_elements=1000,
        M=32,
        ef_construction=100,
        random_seed=123
    )
    assert index.size() == 0
    assert index.dimension() == 64
    assert index.capacity() == 1000


def test_add_single_vector():
    """Test adding a single vector."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(dimension=4)

    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    index.add(1, vec)

    assert index.size() == 1
    assert index.contains(1)
    assert not index.contains(2)


def test_add_multiple_vectors():
    """Test adding multiple vectors."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(dimension=4)

    for i in range(100):
        vec = np.random.randn(4).astype(np.float32)
        index.add(i, vec)

    assert index.size() == 100
    for i in range(100):
        assert index.contains(i)


def test_search_basic():
    """Test basic search functionality."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(dimension=4)

    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    index.add(42, vec)

    ids, dists = index.search(vec, 1)
    assert len(ids) == 1
    assert len(dists) == 1
    assert ids[0] == 42
    assert dists[0] < 1e-6  # Should be ~0 for exact match


def test_search_knn():
    """Test k-nearest neighbor search."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(dimension=4)

    # Add 10 vectors
    for i in range(10):
        vec = np.array([float(i), 0.0, 0.0, 0.0], dtype=np.float32)
        index.add(i, vec)

    # Search for 5 nearest to [5, 0, 0, 0]
    query = np.array([5.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ids, dists = index.search(query, 5)

    assert len(ids) == 5
    assert len(dists) == 5
    assert ids[0] == 5  # Exact match should be first
    assert dists[0] < 1e-6


def test_search_returns_numpy_arrays():
    """Test that search returns numpy arrays."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(dimension=4)

    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    index.add(1, vec)

    ids, dists = index.search(vec, 1)
    assert isinstance(ids, np.ndarray)
    assert isinstance(dists, np.ndarray)
    assert ids.dtype == np.uint64
    assert dists.dtype == np.float32


def test_get_vector():
    """Test retrieving a stored vector."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(dimension=4)

    original = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    index.add(42, original)

    retrieved = index.get_vector(42)
    assert isinstance(retrieved, np.ndarray)
    assert retrieved.dtype == np.float32
    assert len(retrieved) == 4
    np.testing.assert_array_almost_equal(retrieved, original)


def test_ef_search():
    """Test setting and getting ef_search parameter."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(dimension=4)

    # Default should be reasonable
    default_ef = index.get_ef_search()
    assert default_ef > 0

    # Set new value
    index.set_ef_search(100)
    assert index.get_ef_search() == 100


def test_save_load(tmp_path):
    """Test saving and loading an index."""
    import quiverdb_py

    # Create and populate index
    index = quiverdb_py.HNSWIndex(dimension=4)
    vectors = {}
    for i in range(10):
        vec = np.random.randn(4).astype(np.float32)
        vectors[i] = vec
        index.add(i, vec)

    # Save to file
    filepath = str(tmp_path / "test_index.bin")
    index.save(filepath)
    assert os.path.exists(filepath)

    # Load from file
    loaded = quiverdb_py.HNSWIndex.load(filepath)

    # Verify loaded index
    assert loaded.size() == 10
    assert loaded.dimension() == 4

    for i in range(10):
        assert loaded.contains(i)
        retrieved = loaded.get_vector(i)
        np.testing.assert_array_almost_equal(retrieved, vectors[i])


def test_save_load_search_consistency(tmp_path):
    """Test that search results are consistent after save/load."""
    import quiverdb_py

    # Create and populate index
    index = quiverdb_py.HNSWIndex(dimension=8)
    np.random.seed(42)
    for i in range(100):
        vec = np.random.randn(8).astype(np.float32)
        index.add(i, vec)

    # Search before save
    query = np.random.randn(8).astype(np.float32)
    ids_before, dists_before = index.search(query, 10)

    # Save and load
    filepath = str(tmp_path / "test_index.bin")
    index.save(filepath)
    loaded = quiverdb_py.HNSWIndex.load(filepath)

    # Search after load
    ids_after, dists_after = loaded.search(query, 10)

    # Results should match
    np.testing.assert_array_equal(ids_before, ids_after)
    np.testing.assert_array_almost_equal(dists_before, dists_after)


def test_dimension_mismatch_add():
    """Test that adding wrong dimension vector raises error."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(dimension=4)

    wrong_dim = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 3 instead of 4
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        index.add(1, wrong_dim)


def test_dimension_mismatch_search():
    """Test that searching with wrong dimension query raises error."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(dimension=4)

    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    index.add(1, vec)

    wrong_query = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 3 instead of 4
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        index.search(wrong_query, 1)


def test_2d_array_add_raises():
    """Test that adding a 2D array raises error."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(dimension=4)

    vec_2d = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="1-dimensional"):
        index.add(1, vec_2d)


def test_cosine_metric():
    """Test COSINE distance metric."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(
        dimension=4,
        metric=quiverdb_py.HNSWDistanceMetric.COSINE
    )

    # Same direction vectors should have distance ~0
    vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Same direction, different magnitude

    index.add(1, vec1)
    index.add(2, vec2)

    ids, dists = index.search(vec1, 2)
    # Both should have very small distances (cosine distance of parallel vectors)
    assert dists[0] < 0.01
    assert dists[1] < 0.01


def test_dot_metric():
    """Test DOT product metric."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(
        dimension=4,
        metric=quiverdb_py.HNSWDistanceMetric.DOT
    )

    vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)

    index.add(1, vec1)
    index.add(2, vec2)

    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ids, dists = index.search(query, 2)

    # vec2 has larger dot product, should be first (negated in distance)
    assert ids[0] == 2


def test_large_scale():
    """Test with larger number of vectors."""
    import quiverdb_py
    index = quiverdb_py.HNSWIndex(dimension=128, max_elements=10000)

    np.random.seed(42)
    for i in range(1000):
        vec = np.random.randn(128).astype(np.float32)
        index.add(i, vec)

    assert index.size() == 1000

    # Search should work
    query = np.random.randn(128).astype(np.float32)
    ids, dists = index.search(query, 10)

    assert len(ids) == 10
    assert len(dists) == 10
    # Results should be sorted by distance
    for i in range(len(dists) - 1):
        assert dists[i] <= dists[i + 1]


### VectorStore Tests ###

def test_vector_store_import():
    """Test that VectorStore can be imported."""
    import quiverdb_py
    assert hasattr(quiverdb_py, 'VectorStore')
    assert hasattr(quiverdb_py, 'DistanceMetric')


def test_vector_store_basic():
    """Test basic VectorStore operations."""
    import quiverdb_py
    store = quiverdb_py.VectorStore(dimension=4)

    vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    store.add(1, vec)

    assert store.size() == 1
    assert store.contains(1)

    retrieved = store.get(1)
    assert retrieved is not None
    np.testing.assert_array_almost_equal(retrieved, vec)


def test_vector_store_search():
    """Test VectorStore search."""
    import quiverdb_py
    store = quiverdb_py.VectorStore(dimension=4, metric=quiverdb_py.DistanceMetric.L2)

    store.add(1, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    store.add(2, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

    query = np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32)
    ids, dists = store.search(query, 1)

    assert ids[0] == 1


def test_vector_store_remove():
    """Test VectorStore remove operation."""
    import quiverdb_py
    store = quiverdb_py.VectorStore(dimension=4)

    store.add(1, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert store.size() == 1

    store.remove(1)
    assert store.size() == 0
    assert not store.contains(1)


### MMapVectorStore Tests ###

def test_mmap_store_import():
    """Test that MMapVectorStore can be imported."""
    import quiverdb_py
    assert hasattr(quiverdb_py, 'MMapVectorStore')
    assert hasattr(quiverdb_py, 'MMapVectorStoreBuilder')


def test_mmap_store_build_and_load(tmp_path):
    """Test building and loading an mmap store."""
    import quiverdb_py

    filepath = str(tmp_path / "test_mmap.bin")

    # Build
    builder = quiverdb_py.MMapVectorStoreBuilder(dimension=4)
    vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    builder.add(10, vec1)
    builder.add(20, vec2)
    builder.save(filepath)

    # Load
    store = quiverdb_py.MMapVectorStore(filepath)
    assert store.size() == 2
    assert store.dimension() == 4
    assert store.contains(10)
    assert store.contains(20)


def test_mmap_store_search(tmp_path):
    """Test MMapVectorStore search."""
    import quiverdb_py

    filepath = str(tmp_path / "test_mmap_search.bin")

    builder = quiverdb_py.MMapVectorStoreBuilder(dimension=4)
    builder.add(1, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    builder.add(2, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    builder.save(filepath)

    store = quiverdb_py.MMapVectorStore(filepath)
    query = np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32)
    ids, dists = store.search(query, 1)

    assert ids[0] == 1


def test_mmap_store_zero_copy_get(tmp_path):
    """Test that MMapVectorStore get returns zero-copy array."""
    import quiverdb_py

    filepath = str(tmp_path / "test_mmap_zerocopy.bin")

    builder = quiverdb_py.MMapVectorStoreBuilder(dimension=4)
    original = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    builder.add(42, original)
    builder.save(filepath)

    store = quiverdb_py.MMapVectorStore(filepath)
    retrieved = store.get(42)

    assert retrieved is not None
    np.testing.assert_array_almost_equal(retrieved, original)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
