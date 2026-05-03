
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

def get_logger(name='modal_uq'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        h.setFormatter(fmt)
        logger.addHandler(h)
    else:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.stream = sys.stdout
    return logger
