import numpy as np
import pandas as pd
import dask.array as da

class ArraysToDFNeighbors():

    def __init__(self, dataset, shift, axis):
        self.dataset = dataset
        self.shift = shift
        self.axis = axis

    def _lazy_transform_cpu(self):

        overlap = {2: self.shift}

        # Shift the Dask array along the specified axis
        #shifted_dask_cube = da.map_blocks(lambda x: np.roll(x, shift=self.shift, axis=self.axis), self.dataset)
        shifted_dask_cube = da.roll(cube, shift=self.shift, axis=self.axis)

        # Compute the result
        return(shifted_dask_cube.compute())

    def _transform_cpu(self):
        return(np.roll(self.dataset, shift=self.shift, axis=self.axis))


# # IMPLEMENTAR PARA VERSÃO DASK 
# class Neighbors(Transform):

#     def __init__(self, data, x: int = 0, y: int = 0, z: int = 0):
#         self.data = data
#         self.x = x
#         self.y = y
#         self.z = z

#     def _lazy_transform_cpu(self):
#         data = da.array(self.data)
#         x = self.x
#         y = self.y
#         z = self.z
                    
#         x_neighbor = dataset[max(i-x, 0):min(i+x+1, dataset.shape[0]), j, k]
#         x_neighbor = da.delete(x_neighbor, x)

#         y_neighbor = dataset[i, max(j-y, 0):min(j+y+1, dataset.shape[1]), k]
#         y_neighbor = da.delete(y_neighbor, y)

#         z_neighbor = dataset[i, j, max(k-z, 0):min(k+z+1, dataset.shape[2])]
#         z_neighbor = da.delete(z_neighbor, z)
        
#         neighbors = da.concatenate(([data[i, j, k]], x_neighbor.flatten(), y_neighbor.flatten(), z_neighbor.flatten()))
#         return pd.DataFrame(neighbors)

#     def _transform_cpu(self):
#         data = np.array(self.data)
#         x = self.x
#         y = self.y
#         z = self.z
                    
#         x_neighbor = dataset[max(i-x, 0):min(i+x+1, dataset.shape[0]), j, k]
#         x_neighbor = np.delete(x_neighbor, x)

#         y_neighbor = dataset[i, max(j-y, 0):min(j+y+1, dataset.shape[1]), k]
#         y_neighbor = np.delete(y_neighbor, y)

#         z_neighbor = dataset[i, j, max(k-z, 0):min(k+z+1, dataset.shape[2])]
#         z_neighbor = np.delete(z_neighbor, z)
        
#         neighbors = np.concatenate(([data[i, j, k]], x_neighbor.flatten(), y_neighbor.flatten(), z_neighbor.flatten()))
#         return pd.DataFrame(neighbors)
    
#     #def transform(data, x, y, z):

#         # # Inicializa nova numpy array
#         # neighbors = np.zeros([data.shape[0], data.shape[1], data.shape[2]])

#         # for i in range(data.shape[0]):
#         #     for j in range(data.shape[1]):
#         #         for k in range(data.shape[2]):
                    
#         #             # Seleciona vizinhos direita e esquerda e concatena - Dim X
#         #             x_shape          = dataset.shape[0] - 1
#         #             x_neighbor_left  = dataset[max(i-x, 0)      :max(i-1, 0),       j, k]
#         #             x_neighbor_right = dataset[min(i+1, x_shape):min(i+x, x_shape), j, k]
#         #             x_neighbor       = np.concatenate(x_neighbor_left.flatten(), x_neighbor_right.flatten())

#         #             # Seleciona vizinhos direita e esquerda e concatena - Dim Y
#         #             y_shape          = dataset.shape[1] - 1
#         #             y_neighbor_left  = dataset[i, max(j-y, 0)      :max(j-1, 0),     , k]
#         #             y_neighbor_right = dataset[i, min(j+1, y_shape):min(j+y, y_shape), k]
#         #             y_neighbor       = np.concatenate(y_neighbor_left.flatten(), y_neighbor_right.flatten())

#         #             # Seleciona vizinhos direita e esquerda e concatena - Dim Z
#         #             z_shape          = dataset.shape[2] - 1
#         #             z_neighbor_left  = dataset[i, j, max(k-z, 0)      :max(k-1, 0),     ]
#         #             z_neighbor_right = dataset[i, j, min(k+1, z_shape):min(k+z, z_shape)]
#         #             z_neighbor       = np.concatenate(z_neighbor_left.flatten(), z_neighbor_right.flatten())
                    
#         #             # Concatena ponto principal e vizinhos e adiciona à matriz
#         #             neighbor_values    = np.concatenate([data[i, j, k]], x_neighbor, y_neighbor, z_neighbor)
#         #             neighbors[i, j, k] = neighbor_values

#         # return pd.DataFrame(neighbors)

#         # ------------------------------------------



def main():
    dataset = np.array([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
    ])

    dask_cube = da.from_array(dataset, chunks=(3, 3, 3))

    df_neighbors = ArraysToDFNeighbors(dataset=dask_cube, shift=2, axis=0)
    print(df_neighbors._lazy_transform_cpu())



# class ArraysToDFNeighbors(Transform):
    
#     def __init__(self, name: str, data_path: str, x: int, y: int, z: int, side: str, chunks: str = "32Mb"):
#         super().__init__(name=name)
#         self.data_path = data_path
#         self.chunks = chunks

#         self.x = x
#         self.y = y
#         self.z = z
#         self.side = side

#     def __transform_generic(self, dataset, x, y, z, side):
#         # DIM X
#         if x is not None:
#             if side == "right":
#                 # New cube x generic (Right neighbor)
#                 neighbors[:-x, :, :] = dataset[x:, :, :]
#                 neighbors[-x:, :, :] = dataset[-1, :, :]

#             else:
#                 # New cube x generic (Left neighbor)
#                 neighbors[x:, :, :] = dataset[:-x, :, :]
#                 neighbors[:x, :, :] = dataset[0, :, :]


#         # DIM Y
#         if y is not None:
#             if side == "right"
#                 # New cube x generic (Right neighbor)
#                 neighbors[:, :-y, :] = dataset[:, y:, :]
#                 neighbors[:, -y:, :] = dataset[:, -1, :]

#             else:
#                 # New cube x generic (Left neighbor)
#                 neighbors[:, y:, :] = dataset[:, :-y, :]
#                 neighbors[:, :y, :] = dataset[:, 0, :]


#         # DIM Z
#         if z is not None:
#             if side == "right"
#                 # New cube x generic (Right neighbor)
#                 neighbors[:, :, :-z] = dataset[:, :, z:]
#                 neighbors[:, :, -z:] = dataset[:, :, -1]

#             else:
#                 # New cube x generic (Left neighbor)
#                 neighbors[:, :, z:] = dataset[:, :, :-z]
#                 neighbors[:, :, :z] = dataset[:, :, 0]


#         return neighbors

#     def _lazy_transform_cpu(self):
#         data = da.from_zarr(self.data_path, chunks=self.chunks)
#         return self.__transform_generic(data, self.x, self.y, self.z, self.side)

#     def _transform_cpu(self):
#         data = np.load(self.data_path)
#         return self.__transform_generic(data, self.x, self.y, self.z, self.side)