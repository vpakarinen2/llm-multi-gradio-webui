from __future__ import annotations

import os

from app.ui import build_interface
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    demo = build_interface()

    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    demo.launch(server_name=server_name, server_port=server_port, show_error=True)


if __name__ == "__main__":
    main()
