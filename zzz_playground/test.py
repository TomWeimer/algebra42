import numpy as np

def test_basic_contiguous_view():
    Array25 = np.full((4, 4), 10, dtype=np.int64)
    v2 = Array25[1:2, 1:4]
    
    print(f"basic view a25 1 ok: {v2[0, 0] == 10}")
     
    
test_basic_contiguous_view()