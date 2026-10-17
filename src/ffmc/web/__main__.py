"""Run the playground: ``python -m ffmc.web`` (dev server)."""
from __future__ import annotations

import os

from ffmc.web import create_app

if __name__ == "__main__":
    app = create_app()
    app.run(
        host=os.environ.get("FFMC_HOST", "127.0.0.1"),
        port=int(os.environ.get("FFMC_PORT", "5000")),
        debug=bool(os.environ.get("FFMC_DEBUG", "1") == "1"),
    )
