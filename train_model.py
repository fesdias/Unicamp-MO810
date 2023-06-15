import time
import argparse
import numpy as np
import pandas as pd
import dask.array as da
import matplotlib.pyplot as plt

from pathlib import Path
from typing  import Callable, Tuple

from dasf_seismic.attributes.complex_trace import Envelope, InstantaneousFrequency, CosineInstantaneousPhase
from dasf.ml.preprocessing.standardscaler  import StantardScaler
from dasf.ml.xgboost.xgboost               import XGBRegressor
from dasf.transforms                       import ArraysToDataFrame, PersistDaskData, Transform
from dasf.datasets                         import Dataset
from dasf.pipeline                         import Pipeline
from dasf.pipeline.executors               import DaskPipelineExecutor
from dasf.utils.decorators                 import task_handler


# IMPLEMENTAR PARA VERSÃO DASK 
class Neighbors(Transform):
    
    def transform(self, data, x, y, z):
        # (1 + 2*x + 2*y + 2*z)
        neighbors = np.zeros([data.shape[0], data.shape[1], data.shape[2]])

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    
                    # REMOVER PONTO DO MEIO
                    x_neighbor = dataset[max(i-x, 0):min(i+x+1, dataset.shape[0]), j, k]
                    y_neighbor = dataset[i, max(j-y, 0):min(j+y+1, dataset.shape[1]), k]
                    z_neighbor = dataset[i, j, max(k-z, 0):min(k+z+1, dataset.shape[2])]
                    
                    neighbor_values    = np.concatenate(([data[i, j, k]], x_neighbor.flatten(), y_neighbor.flatten(), z_neighbor.flatten()))
                    neighbors[i, j, k] = neighbor_values

        return neighbors


class MyDataset(Dataset):
    # Classe para carregar dados de um arquivo .npy
    def __init__(self, name: str, data_path: str, chunks: str = "32Mb"):
        super().__init__(name=name)
        self.data_path = data_path
        self.chunks = chunks
        
    def _lazy_load_cpu(self):
        return da.from_zarr(self.data_path, chunks=self.chunks)
    
    def _load_cpu(self):
        return np.load(self.data_path)
    
    @task_handler
    def load(self):
        ...


def create_executor(address: str=None) -> DaskPipelineExecutor:
    
    if address is not None:
        addr = ":".join(address.split(":")[:2])
        port = str(address.split(":")[-1])
        print(f"Criando executor. Endereço: {addr}, porta: {port}")
        return DaskPipelineExecutor(local=False, use_gpu=False, address=addr, port=port)

    else:
        return DaskPipelineExecutor(local=True, use_gpu=False)

# FINALIZAR CRIAÇÃO PIPELINE
def create_pipeline(dataset_path: str, attribute_str: str, x: int, y: int, z: int, executor: DaskPipelineExecutor, pipeline_save_location: str = None) -> Tuple[Pipeline, Callable]:
    # Cria o pipeline DASF para ser executado
    print("Criando pipeline....")

    # Declarando os operadores necessários
    dataset        = MyDataset(name="F3 dataset", data_path=dataset_path)
    standardscaler = StantardScaler()
    df_neighbors   = Neighbors()
    #arrays2df      = ArraysToDataFrame()
    persist        = PersistDaskData()
    xgboost        = XGBRegressor()

    if attribute_str == "ENVELOPE":
        attribute = Envelope()
        print("Attribute: Envelope")

    elif attribute_str == "INST-FREQ":
        attribute = InstantaneousFrequency()
        print("Attribute: Inst-Freq")

    else:
        attribute = CosineInstantaneousPhase()
        print("Attribute: Cos-Inst-Phase")
    
    # Compondo o pipeline
    pipeline = Pipeline(
        name = f"F3 seismic {attribute_str}",
        executor = executor)

    pipeline.add(dataset)
    pipeline.add(attribute, X=dataset)
    pipeline.add(standardscaler.fit_transform, X=dataset)
    pipeline.add(df_neighbors, data=standardscaler, x=x, y=y, z=z)
    #pipeline.add(arrays2df, dataset=dataset, envelope=envelope, phase=phase)
    pipeline.add(persist, X=df_neighbors)
    pipeline.add(xgboost.fit, X=persist)
    #pipeline.add(xgboost.predict, X=persist)
    pipeline_save_location = "pipeline.png"

    if pipeline_save_location is not None:
    	pipeline.visualize(filename=pipeline_save_location)
    
    # Retorna o pipeline e o operador kmeans, donde os resultados serão obtidos
    return pipeline, xgboost.fit


def run(pipeline: Pipeline, last_node: Callable) -> np.ndarray:

    print("Executando pipeline")
    start = time.time()
    pipeline.run()
    res = pipeline.get_result_from(last_node)
    res = res.compute()
    end = time.time()
    
    print(f"Feito! Tempo de execução: {end - start:.2f} s")
    return res
    
# EXECUTAR LOCALMENTE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline")
    parser.add_argument("--attribute",      type=str, required=True, help="Nome do atributo a ser usado para treinar o modelo")
    parser.add_argument("--data",           type=str, required=True, help="Caminho para o arquivo .zarr")
    parser.add_argument("--samples-window", type=str, default=0,     help="Número de vizinhos na dimensão das amostras de um traço")
    parser.add_argument("--trace-window",   type=str, default=0,     help="Número de vizinhos na dimensão dos traços de uma inline")
    parser.add_argument("--inline-window",  type=str, default=0,     help="Número de vizinhos na dimensão das inlines")
    parser.add_argument("--address",        type=str, required=True, help="Endereço do dask scheduler. Formato: HOST:PORT")
    parser.add_argument("--output",         type=str, required=True, help="Nome do arquivo de saída onde deve ser gravado o modelo treinado .json")
    args = parser.parse_args()
   
    # Criamos o executor
    executor = create_executor(args.address)

    # Depois o pipeline
    pipeline, last_node = create_pipeline(args.data, artgs.attribute, args.samples-window, args.trace-window, args.inline-window, executor, pipeline_save_location=args.output)

    # Executamos e pegamos o resultado
    res = run(pipeline, last_node)
    print(f"O resultado é um array com o shape: {res.shape}")
    
    # Podemos fazer o reshape e printar a primeira inline
    if args.save_inline_fig is not None:
        res = res.reshape((401, 701, 255))
        import matplotlib.pyplot as plt
        plt.imsave(args.save_inline_fig, res[0], cmap="viridis")
        print(f"Figura da inline 0 salva em {args.save_inline_fig}")
    
