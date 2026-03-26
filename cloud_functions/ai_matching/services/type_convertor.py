import numpy as np
def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to Python native types for MongoDB compatibility.
    """
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def prepare_document_for_mongodb(document):
    """
    Prepare a document for MongoDB insertion by converting NumPy types.
    """
    return convert_numpy_types(document)