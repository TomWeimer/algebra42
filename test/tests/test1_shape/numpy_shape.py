import data.testData as data

def logShape(testData, name, write_log_fn: callable):
    toLog = f"{name}\n dim: {testData.dim}, total_element: {testData.total_element}, shape: {testData.shape}"
    write_log_fn(toLog)
    
def run_tests(write_log_fn: callable):
    logShape(data.scalar, "scalar", write_log_fn)
    logShape(data.empty_vector, "empty", write_log_fn)
    logShape(data.vector_1, "vector", write_log_fn)
    logShape(data.matrix_1, "matrix", write_log_fn)
    logShape(data.row_matrix, "row", write_log_fn)
    logShape(data.column_matrix, "column", write_log_fn)
    logShape(data.array_3D_1, "array3D 1", write_log_fn)
    logShape(data.array_3D_2, "array3D 2", write_log_fn)
    logShape(data.ragged_array, "raggedArray", write_log_fn)
