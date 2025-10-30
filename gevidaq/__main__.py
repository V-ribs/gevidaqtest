import logging
import pathlib
import sys
import threading

# set up logging first
logfile = pathlib.Path(f"./{__package__}.log")
try:
    logfile.replace(f"{logfile}.previous")
except FileNotFoundError:
    pass

file_handler = logging.FileHandler(
    filename=logfile,
)
logging.basicConfig(
    handlers=[file_handler, logging.StreamHandler()],
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
)


def excepthook(*exc_info):
    logging.critical("uncaught exception!", exc_info=exc_info)


def threading_excepthook(exc_info):
    thread = exc_info.thread.name
    logging.critical(
        f"uncaught exception in thread {thread}!", exc_info=exc_info
    )


sys.excepthook = excepthook
threading.excepthook = threading_excepthook


def run():
    from . import Fiumicino  # TODO fix modules misbehaving on import

    Fiumicino.run_app()


if __name__ == "__main__":
    run()

# update check