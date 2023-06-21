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
from dasf.transforms                       import ArraysToDataFrame, Transform
from dasf.pipeline                         import Pipeline
from dasf.datasets                         import Dataset
from dasf.pipeline.executors               import DaskPipelineExecutor
from dasf.utils.decorators                 import task_handler
import dasf.ml.xgboost.xgboost as XGBoost


class Neighbors(Transform):

    def __init__(self, shift, axis):
        self.shift = shift
        self.axis = axis

    def _lazy_transform_cpu(self, dataset):
        return da.roll(dataset, shift=self.shift, axis=self.axis)

    def _transform_cpu(self, dataset):
        return np.roll(dataset, shift=self.shift, axis=self.axis)


class MyDataset(Dataset):
    # Classe para carregar dados de um arquivo .zarr
    def __init__(self, name: str, data_path: str, chunks: str = "63Mb"):
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


def create_pipeline(dataset_path: str, attribute_str: str, x: int, y: int, z: int, executor: DaskPipelineExecutor, pipeline_save_location: str = None) -> Tuple[Pipeline, Callable]:
    # Cria o pipeline DASF para ser executado
    print("Criando pipeline....")

    # Declarando os operadores necessários
    dataset    = MyDataset(name="F3 dataset", data_path=dataset_path)
    xgboost    = XGBoost.XGBRegressor()
    arrays2df1 = ArraysToDataFrame()
    arrays2df2 = ArraysToDataFrame()

    x = int(x)
    y = int(y)
    z = int(z)

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
    pipeline.add(arrays2df1, attribute=attribute)

    i = 0
    dict_n = {}

    if x > 0:
        for j in range(1, x+1):
            neighbor = Neighbors(shift=-j, axis=2)
            pipeline.add(neighbor, dataset=dataset)
            dict_n[f"n_{i}"] = neighbor
            i += 1

            neighbor = Neighbors(shift= j, axis=2)
            pipeline.add(neighbor, dataset=dataset)
            dict_n[f"n_{i}"] = neighbor
            i += 1

    if y > 0:
        for j in range(1, y+1):
            neighbor = Neighbors(shift=-j, axis=1)
            pipeline.add(neighbor, dataset=dataset)
            dict_n[f"n_{i}"] = neighbor
            i += 1

            neighbor = Neighbors(shift= j, axis=1)
            pipeline.add(neighbor, dataset=dataset)
            dict_n[f"n_{i}"] = neighbor
            i += 1

    if z > 0:
        for j in range(1, z+1):
            neighbor = Neighbors(shift=-j, axis=0)
            pipeline.add(neighbor, dataset=dataset)
            dict_n[f"n_{i}"] = neighbor
            i += 1

            neighbor = Neighbors(shift= j, axis=0)
            pipeline.add(neighbor, dataset=dataset)
            dict_n[f"n_{i}"] = neighbor
            i += 1
        
    if x != 0 or y != 0 or z != 0:
        pipeline.add(arrays2df2, dataset=dataset, **dict_n)

    else:
        pipeline.add(arrays2df2, dataset=dataset)
    
    pipeline.add(xgboost.fit, X=arrays2df2, y=arrays2df1)
    
    try:
        pipeline_save_location = "pipeline"
        if pipeline_save_location is not None:
            pipeline.visualize(filename=pipeline_save_location)

    except:
        print("error visualize")

    # Retorna o pipeline e o operador kmeans, donde os resultados serão obtidos
    return pipeline, xgboost.fit


def run(pipeline: Pipeline, last_node: Callable) -> np.ndarray:

    print("Executando pipeline")
    start = time.time()
    pipeline.run()
    res = pipeline.get_result_from(last_node)
    # run model para computar a predição
    #res = res.compute()
    end = time.time()
    
    print(f"Feito! Tempo de execução: {end - start:.2f} s")
    return res
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline")
    parser.add_argument("--attribute",      type=str, required=True, help="Nome do atributo a ser usado para treinar o modelo")
    parser.add_argument("--data",           type=str, required=True, help="Caminho para o arquivo .zarr")
    parser.add_argument("--samples-window", type=str, default=0,     help="Número de vizinhos na dimensão das amostras de um traço")
    parser.add_argument("--trace-window",   type=str, default=0,     help="Número de vizinhos na dimensão dos traços de uma inline")
    parser.add_argument("--inline-window",  type=str, default=0,     help="Número de vizinhos na dimensão das inlines")
    parser.add_argument("--address",        type=str, default=None,  help="Endereço do dask scheduler. Formato: HOST:PORT")
    parser.add_argument("--output",         type=str, required=True, help="Nome do arquivo de saída onde deve ser gravado o modelo treinado .json")
    args = parser.parse_args()
   
    # Criamos o executor
    executor = create_executor(args.address)

    # Depois o pipeline
    pipeline, last_node = create_pipeline(args.data, args.attribute, args.samples_window, args.trace_window, args.inline_window, executor, pipeline_save_location=args.output)

    # Executamos e pegamos o resultado
    res = run(pipeline, last_node)