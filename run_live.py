from __future__ import annotations

import os

import uvicorn

from metal_predictor.live.app import create_app
from metal_predictor.live.forward_bar_runtime import install_forward_bar_runtime
from metal_predictor.live.shadow62_runtime import install_shadow62_runtime


if __name__ == "__main__":
    app = install_forward_bar_runtime(install_shadow62_runtime(create_app()))
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
