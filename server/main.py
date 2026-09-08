from fastapi import FastAPI

if __package__:
    from .api.router import api_router
else:
    # Support `uvicorn main:app` from inside the server directory.
    from api.router import api_router

app = FastAPI(title="Anomaly Dataset API")
app.include_router(api_router)
