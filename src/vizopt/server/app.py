"""FastAPI app serving a `LiveSession` over a WebSocket, plus the frontend."""

import asyncio
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from ..base import OptimConfig, OptimizationProblem, VizOptimizer
from .live import LiveSession
from .protocol import ErrorMessage, client_message_adapter

STATIC_DIR = Path(__file__).parent / "static"
"""Where the built frontend bundle lives (`npm run build` in `frontend/`)."""

_MISSING_FRONTEND_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>vizopt live</title></head>
<body style="font-family: sans-serif; max-width: 40em; margin: 3em auto">
<h1>vizopt live server is running</h1>
<p>The frontend bundle has not been built. From the repository root, run:</p>
<pre>cd frontend
npm install
npm run build</pre>
<p>then restart the server. During frontend development, run
<code>npm run dev</code> instead and open the URL it prints.</p>
</body></html>
"""


def create_app(live: LiveSession, static_dir: Path | None = STATIC_DIR) -> FastAPI:
    """Build the app: a WebSocket at `/ws` and the frontend at `/`.

    The app starts the live session's worker thread on startup and stops it
    on shutdown.

    Args:
        live: The live session to serve.
        static_dir: Directory of the built frontend; when it holds no
            `index.html`, `/` serves a page explaining how to build it.

    Returns:
        The FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        live.start()
        try:
            yield
        finally:
            live.stop()

    app = FastAPI(title="vizopt live", lifespan=lifespan)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()
        outbox: list[dict] = []

        def notify() -> None:
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:  # event loop already closed
                pass

        async def send_frames() -> None:
            # The only task that writes to the socket: frames and errors.
            await websocket.send_json(live.hello().model_dump(mode="json"))
            sent_version = -1
            while True:
                while outbox:
                    await websocket.send_json(outbox.pop(0))
                version, frame = live.latest()
                if frame is not None and version != sent_version:
                    await websocket.send_json(frame)
                    sent_version = version
                await wake.wait()
                wake.clear()

        async def receive_commands() -> None:
            while True:
                data = await websocket.receive_json()
                try:
                    message = client_message_adapter.validate_python(data)
                    live.validate(message)
                except (ValidationError, ValueError) as error:
                    outbox.append(ErrorMessage(message=str(error)).model_dump())
                    wake.set()
                    continue
                live.submit(message)

        unsubscribe = live.subscribe(notify)
        tasks = [
            asyncio.create_task(send_frames()),
            asyncio.create_task(receive_commands()),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                error = task.exception()
                if error is not None and not isinstance(error, WebSocketDisconnect):
                    raise error
        finally:
            unsubscribe()
            for task in tasks:
                task.cancel()

    if static_dir is not None and (static_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    else:

        @app.get("/", response_class=HTMLResponse)
        async def missing_frontend() -> str:
            return _MISSING_FRONTEND_PAGE

    return app


def serve(
    optimizer: VizOptimizer | OptimizationProblem,
    optim_config: OptimConfig | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    steps_per_frame: int = 10,
    fps: float = 30.0,
) -> None:
    """Run an optimization live in the browser; blocks until interrupted.

    Starts a fresh session, serves it at `http://host:port`, and steps it
    while clients watch and steer: drag elements to pin and move them, pause,
    reheat, change weights. The problem must have a `scene_configuration`.

    Example:
        from vizopt.server import serve
        serve(LayeredGraphOptimizer(graph), OptimConfig(learning_rate=3e-3))

    Args:
        optimizer: A `VizOptimizer` (its problem is built here) or an
            already-built `OptimizationProblem`.
        optim_config: Optimizer settings; `n_iters` is the length of each
            learning-rate decay, after which the run settles until the next
            interaction.
        host: Interface to bind; the default only accepts local connections.
        port: Port to listen on.
        open_browser: Open the frontend in the default browser.
        steps_per_frame: Optimization steps between two frames.
        fps: Target frames per second while running.
    """
    session = optimizer.session(optim_config)
    live = LiveSession(session, steps_per_frame=steps_per_frame, fps=fps)
    app = create_app(live)
    url = f"http://{host}:{port}/"
    print(f"vizopt live: serving at {url} (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=[url]).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
