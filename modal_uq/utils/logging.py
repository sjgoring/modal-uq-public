
import logging
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class _TimestampedTee:
    def __init__(self, stream, log_handle, source_label):
        self._stream = stream
        self._log_handle = log_handle
        self._source_label = source_label
        self._buffer = ""

    def write(self, text):
        if not text:
            return 0
        self._stream.write(text)
        self._buffer += text.replace("\r\n", "\n")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._write_line(line)
        return len(text)

    def flush(self):
        if self._buffer:
            self._write_line(self._buffer)
            self._buffer = ""
        self._stream.flush()
        self._log_handle.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self):
        return getattr(self._stream, "fileno")()

    def _write_line(self, line):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_handle.write(f"{timestamp} | {self._source_label} | {line}\n")
        self._log_handle.flush()


@contextmanager
def capture_stdout_stderr(log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "a", encoding="utf-8")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = _TimestampedTee(old_stdout, log_handle, "STDOUT")
    sys.stderr = _TimestampedTee(old_stderr, log_handle, "STDERR")
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_handle.flush()
        log_handle.close()

def get_logger(name='modal_uq', log_path=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    terminal_stream = getattr(sys.stdout, "_stream", sys.stdout)

    stream_handler = None
    file_handler = None

    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            if log_path is not None and Path(handler.baseFilename) == Path(log_path):
                file_handler = handler
                file_handler.setFormatter(fmt)
            else:
                logger.removeHandler(handler)
                handler.close()
        elif isinstance(handler, logging.StreamHandler):
            stream_handler = handler
            stream_handler.stream = terminal_stream
            stream_handler.setFormatter(fmt)

    if stream_handler is None:
        stream_handler = logging.StreamHandler(terminal_stream)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    if log_path is not None and file_handler is None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
