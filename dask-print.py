import json
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
from dasf.transforms                       import ArraysToDataFrame, Transform, PersistDaskData
from dasf.pipeline                         import Pipeline
from dasf.datasets                         import Dataset
from dasf.pipeline.executors               import DaskPipelineExecutor
from dasf.utils.decorators                 import task_handler
import dasf.ml.xgboost.xgboost as XGBoost

from dask.distributed import Client, performance_report


class MyDataset(Dataset):
    # Classe para carregar dados de um arquivo .zarr
    def __init__(self, name: str, data_path: str, chunks: str = "10Mb"):
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


def create_pipeline(dataset_path: str, attribute_str: str, executor: DaskPipelineExecutor) -> Tuple[Pipeline, Callable]:
    # Cria o pipeline DASF para ser executado
    print("Criando pipeline....")

    # Declarando os operadores necessários
    dataset    = MyDataset(name="F3 dataset", data_path=dataset_path)
    arrays2df1 = ArraysToDataFrame()
    persist    = PersistDaskData()

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
    pipeline.add(persist, X=arrays2df1)

    # Retorna o pipeline e o operador kmeans, donde os resultados serão obtidos
    return pipeline, persist


def run(pipeline: Pipeline, last_node: Callable) -> np.ndarray:

    print("Executando pipeline")
    start = time.time()
    pipeline.run()
    res = pipeline.get_result_from(last_node)
    res = res.compute()
    end = time.time()
    
    print(f"Feito! Tempo de execução: {end - start:.2f} s")
    return res
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline")
    parser.add_argument("--attribute",      type=str, required=True, help="Nome do atributo a ser usado para treinar o modelo")
    parser.add_argument("--data",           type=str, required=True, help="Caminho para o arquivo .zarr")
    parser.add_argument("--address",        type=str, default=None,  help="Endereço do dask scheduler. Formato: HOST:PORT")
    args = parser.parse_args()

    # Criamos o executor
    executor = create_executor(args.address)

    # Depois o pipeline
    pipeline, last_node = create_pipeline(args.data, args.attribute, executor)

    # Executamos e pegamos o resultado
    res = run(pipeline, last_node)
    print(f"O resultado é um array com o shape: {res.shape}")


    # Podemos fazer o reshape e printar a primeira inline
    inline_name = f"inlines/{args.attribute}_inline.jpeg"
    res = res.values.reshape((401, 701, 255))
    plt.imsave(inline_name, res[0], cmap="viridis")
    print(f"Figura da inline 0 salva em {inline_name}")

    #docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 dask-print.py --attribute COS-INST-PHASE --data data/F3_train.zarr --address tcp://143.106.16.207:8786
