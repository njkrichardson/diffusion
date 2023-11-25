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

def image_grid(images: List[List[Tensor]], as_ndarray: Optional[bool]=False) -> None: 
    num_rows: int = len(images) 
    num_columns: int = len(images[0]) 
    figure, axs = plt.subplots(figsize=(10, 3 * num_rows), nrows=num_rows, ncols=num_columns, dpi=128, squeeze=False) 

    for row_idx, row in enumerate(images): 
        for column_idx, image in enumerate(row): 
            ax = axs[row_idx, column_idx]
            if isinstance(image, Tensor): 
                image = image.cpu().numpy()
            ax.imshow(image.squeeze(), cmap="Greys") # TODO generalize to different observations
            ax.set_xticks([])
            ax.set_yticks([])

    if as_ndarray: 
        image: ndarray = image_from_figure(figure)
        plt.close()
        return image 
    else: 
        plt.show()

def show_corruption(encoder: callable, x: Tensor, timesteps: Tensor, as_ndarray: Optional[bool]=False) -> Optional[ndarray]: 
    corruptions: List[List[Tensor]] = [] 

    for _x in x: 
        corruptions.append([_x] + [corrupt(encoder, _x, timestep.reshape(1)) for timestep in timesteps])

    num_rows: int = len(corruptions) 
    num_columns: int = len(corruptions[0]) 
    figure, axs = plt.subplots(figsize=(10, 10), nrows=num_rows, ncols=num_columns, dpi=128, squeeze=False) 
    
    for row_idx, row in enumerate(corruptions): 
        for column_idx, image in enumerate(row): 
            ax = axs[row_idx, column_idx]
            ax.imshow(image.cpu().numpy(), cmap="Greys") # TODO generalize to different observations
            ax.set_xticks([])
            ax.set_yticks([])

    if as_ndarray: 
        image: ndarray = image_from_figure(figure)
        plt.close()
        return image 

    else: 
        plt.show()
