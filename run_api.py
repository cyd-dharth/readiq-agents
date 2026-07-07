import os
import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    port = int(os.environ.get("PORT", settings.chat_api_port))
    uvicorn.run(
        "app.server:app",
        host=settings.chat_api_host,
        port=port,
        reload=False,
    )
