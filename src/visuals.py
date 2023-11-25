from io import BytesIO
from typing import List, Optional 

import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np 
import torch 

from constants import ndarray, Tensor
from utils import corrupt 

matplotlib.use("Agg")

def image_from_figure(figure, dpi: Optional[int]=128) -> ndarray: 
    buffer = BytesIO()
    figure.savefig(buffer, format='raw', dpi=dpi)
    buffer.seek(0)
    image: ndarray = np.reshape(np.frombuffer(buffer.getvalue(), dtype=np.uint8), newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1))
    buffer.close()
    return image

def show_corruption(encoder: callable, x: Tensor, timesteps: Tensor, as_ndarray: Optional[bool]=False) -> Optional[ndarray]: 
    corruptions: List[List[Tensor]] = [[x] + [corrupt(encoder, x, timestep.reshape(1)) for timestep in timesteps]]
    num_rows: int = len(corruptions) 
    num_columns: int = len(corruptions[0]) 
    figure, axs = plt.subplots(figsize=(10, 10), nrows=num_rows, ncols=num_columns, dpi=128) 
    
    for column_idx, instance in enumerate(corruptions[0]): 
        ax = axs[column_idx]
        ax.scatter(0, instance[0]) # TODO generalize to different observations

    if as_ndarray: 
        image: ndarray = image_from_figure(figure)
        plt.close()
        return image 

    else: 
        plt.show()
