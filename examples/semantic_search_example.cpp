#include "core/vector_store.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <iomanip>

// Simple example demonstrating semantic search with quiverdb
int main() {
  std::cout << "=== QuiverDB Semantic Search Example ===\n\n";

  // Create a vector store for 384-dimensional embeddings (common for smaller models)
  const size_t dim = 384;
  quiverdb::VectorStore store(dim, quiverdb::DistanceMetric::COSINE);

  // Simulate document embeddings (in practice, these would come from an embedding model)
  std::mt19937 gen(42);
  std::normal_distribution<float> dis(0.0f, 1.0f);

  // Document metadata (IDs map to documents)
  std::vector<std::string> documents = {
      "The quick brown fox jumps over the lazy dog",
      "Machine learning is a subset of artificial intelligence",
      "QuiverDB is a fast vector database for edge devices",
      "C++ is a powerful programming language",
      "Vector databases enable semantic search capabilities"
  };

  std::cout << "Adding " << documents.size() << " document embeddings to the store...\n";

  // Add document vectors to the store
  for (size_t i = 0; i < documents.size(); ++i) {
    std::vector<float> embedding(dim);

    // Generate random embedding (normally you'd use a real embedding model)
    // For demo purposes, we add some structure:
    // - Documents about ML/AI have higher values in first half
    // - Documents about programming have higher values in second half
    for (size_t j = 0; j < dim; ++j) {
      float value = dis(gen);

      // Artificial structure for demo
      if (i == 1 || i == 4) { // ML/AI docs
        if (j < dim / 2) value += 0.5f;
      } else if (i == 2 || i == 3) { // Programming docs
        if (j >= dim / 2) value += 0.5f;
      }

      embedding[j] = value;
    }

    // Normalize for cosine similarity
    float norm = 0.0f;
    for (float val : embedding) {
      norm += val * val;
    }
    norm = std::sqrt(norm);
    for (float& val : embedding) {
      val /= norm;
    }

    store.add(i, embedding.data());
  }

  std::cout << "Store size: " << store.size() << " vectors\n\n";

  // Create query embeddings
  std::cout << "=== Query 1: AI/ML related ===\n";
  {
    std::vector<float> query(dim);
    for (size_t j = 0; j < dim; ++j) {
      float value = dis(gen);
      if (j < dim / 2) value += 0.5f; // ML-like pattern
      query[j] = value;
    }

    // Normalize
    float norm = 0.0f;
    for (float val : query) norm += val * val;
    norm = std::sqrt(norm);
    for (float& val : query) val /= norm;

    // Search for top 3 results
    auto results = store.search(query.data(), 3);

    std::cout << "Top 3 results:\n";
    for (size_t i = 0; i < results.size(); ++i) {
      std::cout << std::fixed << std::setprecision(4)
                << "  " << (i + 1) << ". [Distance: " << results[i].distance << "] "
                << documents[results[i].id] << "\n";
    }
  }

  std::cout << "\n=== Query 2: Programming related ===\n";
  {
    std::vector<float> query(dim);
    for (size_t j = 0; j < dim; ++j) {
      float value = dis(gen);
      if (j >= dim / 2) value += 0.5f; // Programming-like pattern
      query[j] = value;
    }

    // Normalize
    float norm = 0.0f;
    for (float val : query) norm += val * val;
    norm = std::sqrt(norm);
    for (float& val : query) val /= norm;

    // Search for top 3 results
    auto results = store.search(query.data(), 3);

    std::cout << "Top 3 results:\n";
    for (size_t i = 0; i < results.size(); ++i) {
      std::cout << std::fixed << std::setprecision(4)
                << "  " << (i + 1) << ". [Distance: " << results[i].distance << "] "
                << documents[results[i].id] << "\n";
    }
  }

  // Demonstrate other operations
  std::cout << "\n=== Other Operations ===\n";

  // Get a specific vector
  std::cout << "Retrieving document 2...\n";
  const float* vec = store.get(2);
  if (vec) {
    std::cout << "  First 5 dimensions: ";
    for (size_t i = 0; i < 5; ++i) {
      std::cout << std::fixed << std::setprecision(3) << vec[i] << " ";
    }
    std::cout << "...\n";
  }

  // Remove a document
  std::cout << "Removing document 0...\n";
  store.remove(0);
  std::cout << "  Store size after removal: " << store.size() << "\n";

  // Check if document exists
  std::cout << "Checking if document 0 exists: "
            << (store.contains(0) ? "Yes" : "No") << "\n";
  std::cout << "Checking if document 1 exists: "
            << (store.contains(1) ? "Yes" : "No") << "\n";

  std::cout << "\n=== Performance Stats ===\n";
  std::cout << "Vector dimension: " << store.dimension() << "\n";
  std::cout << "Distance metric: ";
  switch (store.metric()) {
    case quiverdb::DistanceMetric::L2:
      std::cout << "L2 (Euclidean)\n";
      break;
    case quiverdb::DistanceMetric::COSINE:
      std::cout << "Cosine\n";
      break;
    case quiverdb::DistanceMetric::DOT:
      std::cout << "Dot Product (MIPS)\n";
      break;
  }

  std::cout << "Total vectors: " << store.size() << "\n";

  return 0;
}
