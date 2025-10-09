import numpy as np


from data.testData import TestDataScalar, TestDataArray


# Obtain the index ( flat index ) from multiple indices
def obtainPos(dim, strides, indices):
    offset = 0

    if dim == 0:
        return 0

    if dim == 1:
        return indices[0]

    for i in range(0, dim):
        offset += indices[i] * strides[i]
    return offset


def obtainStrides(shape, dim):
    # Pre-allocate a tuple of zeros
    strds = [0] * dim
    # The stride for the first dimension is always 1
    strds[0] = 1

    # For subsequent dimensions, stride[i] = stride[i-1] * size[i-1]
    for i in range(1, dim):
        strds[i] = strds[i - 1] * shape[i - 1]
    return tuple(strds)

    # function pretty_index(idx...)
    #     parts = map(pretty_index, idx)
    #     return join(parts, ", ")
    # end

    # function pretty_index(idx::AbstractArray)
    #     parts = map(v ->
    #         v === Colon() ? ":" :
    #         string(v),
    #         idx
    #     )
    #     return "[" * join(parts, ", ") * "]"
    # end

    # function pretty_index(idx::Tuple)
    #     parts = map(v ->
    #         v === Colon() ? ":" :
    #         isa(v, AbstractArray) ? pretty_index(v) :
    #         string(v),
    #         idx
    #     )
    #     return "(" * join(parts, ", ") * ")"
    # end

    # function pretty_index(idx::Colon)
    #     return ":"
    # end

    # function pretty_index(idx::Number)
    #     return string(idx)
    # end




class PrettyIndex:

    @staticmethod
    def formatIndex(*idx):
        """Format indices, handling slices, tuples, numbers."""
        parts = []
        for i in idx:
            if isinstance(i, slice):
                parts.append(":")
            else:
                # Convert to 1-based index for printing
                parts.append(str(i + 1) if isinstance(i, int) else str(i))
        return "(" + ", ".join(parts) + ")"

    @staticmethod
    def format(*idx):
        """Format indices, handling slices, tuples, numbers."""
        parts = []
        for i in idx:
            if isinstance(i, slice):
                parts.append(":")
            else:
                # Convert to 1-based index for printing
                parts.append(
                    str(i + 1)
                    if isinstance(i, int) or isinstance(i, np.int64)
                    else str(i)
                )
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts)


def toJuliaCoordinates(indices):
    if len(indices) <= 1:
        return indices
    elif len(indices) == 2:
        return (indices[1], indices[0])
    else:
        tmp_array = [x for x in indices]
        end = len(tmp_array) - 1
        tmp = tmp_array[end - 1]
        tmp_array[end - 1] = tmp_array[end]
        tmp_array[end] = tmp
        return tuple(reversed(tmp_array))


def logShape_testData(testData, write_log_fn: callable):
    logShape(testData.dim, testData.total_element, testData.shape, write_log_fn)


def logShape(dim, total_element, shape, write_log_fn: callable):
    toLog = f"dim: {dim}\ntotal_element: {total_element}\nshape: {shape}"
    write_log_fn(toLog)


def log_scalar(ndarray: TestDataScalar, write_log_fn: callable):
    write_log_fn(f"{ndarray.data}")


def write_scalar_content(scalar, write_log_fn):
    ndarray = np.array(scalar.data)
    write_log_fn(f"{ndarray[()]}")
    logShape(ndarray.ndim, ndarray.size, ndarray.shape, write_log_fn)


def write_content(ndarray, write_log_fn: callable):
    if isinstance(ndarray, TestDataScalar):
        log_scalar(ndarray, write_log_fn)
    elif isinstance(ndarray, np.ndarray):
        logShape(ndarray.ndim, ndarray.size, ndarray.shape, write_log_fn)
        log_by_index(ndarray, write_log_fn)
        log_by_iterator(ndarray, write_log_fn)
        log_by_slices(ndarray.shape, ndarray, ndarray.ndim, write_log_fn)
    else:
        logShape_testData(ndarray, write_log_fn)
        log_by_index_testData(ndarray, write_log_fn)
        log_by_iterator_testData(ndarray, write_log_fn)
        log_by_slices_testData(ndarray, write_log_fn)


def log_by_index_testData(ndarray: TestDataArray, write_log_fn: callable):
    log_by_index(ndarray.matrix, write_log_fn)

def round_0(n, ndigits=7):
    if isinstance(n, float) and n == -0.0:
        n =  0.0
    elif isinstance(n, int) and n == -0:
        n = 0
    return round(n, ndigits=ndigits)

def log_by_index(ndarray, write_log_fn: callable):
    """Log each element by explicit indices, rounding to 7 digits."""
    content_by_index = ["index: "]
    matrix = np.transpose(ndarray)
    shape = matrix.shape

    for indices in np.ndindex(shape):
        element = matrix[indices]

        # Round element(s) to 7 digits
        if isinstance(element, list):
            x = [round_0(e, 7) for e in element]
        else:
            x = round_0(element, 7)

        content_by_index.append(f"{PrettyIndex.formatIndex(*reversed(indices))}: {x}, ")

    write_log_fn("".join(content_by_index))


def log_by_iterator_testData(ndarray: TestDataArray, write_log_fn: callable):
    log_by_iterator(ndarray.matrix, write_log_fn)


def log_by_iterator(matrix, write_log_fn: callable):
    """Log each element using flat iteration."""
    content_by_iterator = ["iterator: "]
    for val in matrix.flatten(order="F"):

        if isinstance(val, list):
            x = [round_0(e, 7) for e in val]
        else:
            x = round_0(val, 7)
        content_by_iterator.append(f"{x} ")
    write_log_fn("".join(content_by_iterator))


def fortran_ndindex(shape):
    """Generate indices in Fortran (column-major) order."""
    size = np.prod(shape)
    for flat in range(size):
        yield np.unravel_index(flat, shape, order="F")


def log_by_slices_testData(ndarray: TestDataArray, write_log_fn: callable):
    log_by_slices(
        ndarray.matrix.shape, ndarray.matrix, ndarray.matrix.ndim, write_log_fn
    )


def log_by_slices(shape, matrix, ndim, write_log_fn: callable):
    """Log slices along each axis, rounding all elements to 7 digits."""

    write_log_fn("slices:")
    content_by_slices = []

    for axis in range(ndim):
        # create the shape for iterating over fixed indices along other axes
        iter_shape = tuple(1 if i == axis else s for i, s in enumerate(shape))
        for idx in fortran_ndindex(iter_shape):
            # Build a tuple of slice objects for the full array
            indices = tuple(slice(None) if i == axis else idx[i] for i in range(ndim))
            slice_view = matrix[indices]

            # Flatten and round, handling elements that are lists
            rounded_elements = []
            for x in slice_view.flatten():
                if isinstance(x, (list, np.ndarray)):
                    rounded_elements.append([ round_0(e, 7) for e in x])
                else:
                    rounded_elements.append(round_0(x, 7))

            # Convert to string
            slice_str = "[" + ", ".join(str(e) for e in rounded_elements) + "]"
            content_by_slices.append(
                f"array[{PrettyIndex.format(*indices)}] = {slice_str}\n"
            )

    write_log_fn("".join(content_by_slices))

