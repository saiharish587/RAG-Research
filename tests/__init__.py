"""Test suite for the SLM-RAG benchmark.

Every test here runs on the Python standard library alone -- no GPU, no model
weights, no network, no Ollama. Heavy dependencies are injected as stubs so the
suite is runnable on any machine, including CI and the workstation used for
paper editing rather than the benchmark server.

Run with::

    python -m unittest discover -s tests -t .
"""
