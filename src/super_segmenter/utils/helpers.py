import logging


def set_logging_level(verbosity: str):
    logging.basicConfig(
        level=logging._nameToLevel[verbosity],
        format="%(asctime)s %(message)s",
        datefmt="%I:%M:%S %p",
    )


def padding_same(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)
