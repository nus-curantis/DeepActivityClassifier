try:
    from ._dtw_lib import fastdtw, dtw, relaxed_dtw, fast_relaxed_dtw, matrix_convert
except ImportError:
    from .dtw_lib import fastdtw, dtw, relaxed_dtw, fast_relaxed_dtw, matrix_convert
