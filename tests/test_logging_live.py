from modal_uq.utils.logging import capture_stdout_stderr, get_logger


def test_capture_stdout_stderr_flushes_each_line(tmp_path):
    log_path = tmp_path / "run.log"

    with capture_stdout_stderr(str(log_path)):
        print("hello from stdout")
        with open(log_path, encoding="utf-8") as handle:
            contents = handle.read()

    assert "hello from stdout" in contents


def test_get_logger_writes_directly_to_log_file(tmp_path):
    log_path = tmp_path / "run.log"

    with capture_stdout_stderr(str(log_path)):
        logger = get_logger("test_runtime_logger", log_path=str(log_path))
        logger.info("direct logger message")
        with open(log_path, encoding="utf-8") as handle:
            contents = handle.read()

    assert "direct logger message" in contents