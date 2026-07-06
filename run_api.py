import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.server:app",
        host=settings.chat_api_host,
        port=settings.chat_api_port,
        reload=False,
    )
