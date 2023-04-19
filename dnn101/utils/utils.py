import math


def convert_to_base(a: tuple, b: float = 2.0) -> tuple:
    """
    Convert tuple of floats to a base-exponent pair for nice printouts.
    See use in, e.g., :py:func:`hessQuik.utils.input_derivative_check.input_derivative_check`.
    """
    outputs = ()
    for i in range(len(a)):
        if a[i] <= 0:
            # catch case when equal to 0
            c, d = -1, 0
        else:
            d = math.floor(math.log2(a[i]) / math.log2(b))
            c = b ** (math.log2(a[i]) / math.log2(b) - d)

        outputs += (c, d)

    return outputs

