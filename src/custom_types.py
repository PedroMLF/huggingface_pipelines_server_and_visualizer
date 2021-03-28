from typing import Dict, List, Union
from typing import NewType

import numpy as np


# Pipeline related types
FinalPrediction = NewType("FinalPrediction", Dict["str", Union[int, float, str]])
RawPrediction = NewType("RawPrediction", Dict["str", Union[np.int64, np.float64, str]])
