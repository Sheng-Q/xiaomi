# Copyright (C) 2026 Xiaomi Corporation.
"""Top-level package for MiBot.

Keep package import lightweight so runtime-only tools can import
``mibot.server.runtime.client`` without pulling in the training stack
and its optional dependencies (for example ``lightning``).

Training and data symbols remain available from their explicit
subpackages, such as ``mibot.models`` and ``mibot.data``.
"""

__all__: list[str] = []
