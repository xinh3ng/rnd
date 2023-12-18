from gunicorn.app.base import BaseApplication
import multiprocessing
import os
import typer
import uvicorn


cli = typer.Typer()


class StandaloneApplication(BaseApplication):
    def __init__(self, app: str, options: dict = None):
        self.options = options or {}
        self.application = app
        super(StandaloneApplication, self).__init__()

    def load_configs(self):
        configs = dict([(k, v) for k, v in (self.options.items()) if k in self.cfg.settings and v is not None])

        for k, v in configs.items():
            self.cfg.set(k.lower(), v)

    def load(self):
        return self.application


@cli.command("run-api-server")
def run_api_server(
    port: int = 80,
    host: str = "0.0.0.0",
    log_level: str = "info",
    reload: bool = True,
    workers: int = 1,
    # workers: int = 2 * multiprocessing.cpu_count() + 1 as a general suggestion
):
    """Run the API development server (uvicorn)."""
    assert workers >= 1 and isinstance(workers, int)
    module_str = "main:app"

    if workers == 1:
        uvicorn.run(
            module_str,
            host=host,
            port=port,
            log_level=log_level,
            reload=reload,
        )
    else:
        # os.system(f"gunicorn -w {workers} -k uvicorn.workers.UvicornWorker -b {host}:{port} main:app")
        options = {
            "bind": "%s:%s" % (host, port),
            "workers": workers,
            "worker_class": "uvicorn.workers.UvicornWorker",
        }
        StandaloneApplication(module_str, options).run()


@cli.command()
def info():
    pass


if __name__ == "__main__":
    cli()
