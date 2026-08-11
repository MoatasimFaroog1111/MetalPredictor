from __future__ import annotations

import os

import uvicorn

from metal_predictor.live.app import create_app


if __name__ == "__main__":
    uvicorn.run(
        create_app(),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
