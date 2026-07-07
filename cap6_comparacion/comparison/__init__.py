"""Comparison and benchmarking tools for alternative anomaly detection models.

This package contains:
    - alternative_models: Neural network architectures (StandardAE, SparseAE, etc.)
    - classical_models: Classical ML models (Isolation Forest, SVM, LOF, PCA)
    - benchmark: Full benchmark runner for training and comparing all models
    - compare: Visualization of benchmark results

Usage:
    cd bin
    python -m model.comparison.benchmark              # full benchmark
    python -m model.comparison.benchmark --quick       # quick benchmark
    python -m model.comparison.compare                 # generate comparison charts
"""
