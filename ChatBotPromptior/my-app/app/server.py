import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from packages.chain import chain


app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/playground")


add_routes(
    app,
    chain,
    playground_type='default',  # Could also be chat, however seems to be a bug in LangServe
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True
)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(app, host='0.0.0.0', port=port)
