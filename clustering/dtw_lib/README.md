[![Maintainability](https://api.codeclimate.com/v1/badges/5e8517e149abad38c323/maintainability)](https://codeclimate.com/github/nus-curantis/dtw_lib/maintainability)

# DTW Library
### Example
```
from dtw_lib import _dtw_lib
from scipy.spatial.distance import euclidean

x = [(1, 2, 1), (2, 3, 2), (6, 4, 1), (3, 5, 3), (2, 3, 3), (4, 5, 6)] 
y = [(2, 4, 1), (5, 6, 2), (6, 4, 1), (3, 7, 3)]
```

 Classic DTW and Fast DTW

``` 
distance, path, D = _dtw_lib.dtw(x, y,dist=euclidean) #classic

distance, path, D = _dtw_lib.fastdtw(x, y,,dist=euclidean) #fast
``` 
Relaxed versions of Classic DTW and Fast DTW 
``` 
relax = 1

distance, path, D = _dtw_lib.dtw(x, y, relax=relax, dist=euclidean) #classic

distance, path, D = _dtw_lib.fastdtw(x, y, relax=relax, dist=euclidean ) #fast
```

### Cythonize
Enter this command to cythonize the code.

``` 
python setup.py build_ext --inplace
```

#### Note
1. In the above examples D is the distance matrix in dictionary data type, use the code below to convert to 2D numpy array
``` 
from dtw_lib import conversion

D_matrix = conversion.matrix(D)
```

2. Default relax parameter is set to 0

3. dtw_lib is the python version of this code
