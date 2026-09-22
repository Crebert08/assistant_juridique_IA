"""Reusable sys.modules stubs for heavy / uninstalled third-party deps.

Import this module and call ``install_stubs()`` BEFORE importing anything
from ``app``. Every name in ``STUB_MODULES`` is replaced by a ``MagicMock``
so that ``from x.y import Z`` works and module-level code in app/ (PDF
loading, Chroma, embeddings, LLM construction) runs without I/O or network.

Stubbing a module that app/ never imports is harmless, so later phases can
simply append names (e.g. langchain_anthropic, langchain_huggingface).
"""
import sys
from unittest.mock import MagicMock

STUB_MODULES = [
    # Jev / TypeSafe
    "typesafe_sdk",
    # env
    "dotenv",
    # langchain family (phase 1: Google)
    "langchain",
    "langchain.text_splitter",
    "langchain.chains",
    "langchain.chains.combine_documents",
    "langchain_text_splitters",
    "langchain_community",
    "langchain_community.document_loaders",
    "langchain_community.vectorstores",
    "langchain_chroma",
    "langchain_google_genai",
    "langchain_core",
    "langchain_core.prompts",
    "langchain_core.documents",
    "langchain_core.output_parsers",
    "langchain_core.runnables",
    # phase 2: Claude + local HF embeddings
    "anthropic",
    "langchain_anthropic",
    "langchain_huggingface",
    # web layer (not imported by tests, stubbed for safety)
    "fastapi",
    "fastapi.middleware",
    "fastapi.middleware.cors",
    "pydantic",
]

_INSTALLED = {}  # name -> stub installed by this module


def install_stubs(extra=()):
    """Insert MagicMock modules into sys.modules for STUB_MODULES + extra.

    Returns the dict of installed stubs (name -> mock) for inspection.
    Always overwrites real packages so tests are deterministic even if one
    happens to be installed (we never want real PDF/LLM/network work), but
    reuses a stub this module already installed: several test modules call
    install_stubs() and must all see the same objects app/ was imported with.
    """
    installed = {}
    for name in list(STUB_MODULES) + list(extra):
        mod = _INSTALLED.get(name)
        if mod is None or sys.modules.get(name) is not mod:
            mod = MagicMock(name=f"stub:{name}")
            mod.__name__ = name
            mod.__path__ = []  # make it look like a package
            sys.modules[name] = mod
            _INSTALLED[name] = mod
        installed[name] = mod
    # Wire children as attributes of parents so `import a.b` + `a.b.X` works.
    for name, mod in installed.items():
        if "." in name:
            parent, _, child = name.rpartition(".")
            if parent in installed:
                setattr(installed[parent], child, mod)
    return installed
