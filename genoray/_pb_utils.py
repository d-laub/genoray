from contextlib import contextmanager

import polars_bio as pb


@contextmanager
def pb_zero_based(zero_based: bool):
    old_setting = pb.get_option(pb.POLARS_BIO_COORDINATE_SYSTEM_ZERO_BASED)
    pb.set_option(pb.POLARS_BIO_COORDINATE_SYSTEM_ZERO_BASED, str(zero_based).lower())
    try:
        yield
    finally:
        pb.set_option(pb.POLARS_BIO_COORDINATE_SYSTEM_ZERO_BASED, old_setting)
