import time
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import dask.array as da
import matplotlib.pyplot as plt
import dasf.ml.xgboost.xgboost as XGBoost

from pathlib import Path
from typing  import Callable, Tuple

from dasf_seismic.attributes.complex_trace import Envelope, InstantaneousFrequency, CosineInstantaneousPhase
from dasf.ml.preprocessing.standardscaler  import StantardScaler
from dasf.transforms                       import ArraysToDataFrame, Transform, PersistDaskData
from dasf.pipeline                         import Pipeline
from dasf.datasets                         import Dataset
from dasf.pipeline.executors               import DaskPipelineExecutor
from dasf.utils.decorators                 import task_handler

# Implement Dashboard
from dask.distributed import Client, performance_report

# Implement XGBRegressor
import GPUtil
from dasf.transforms import Fit
from dasf.transforms import Predict
from dasf.transforms import FitPredict

class XGBRegressor(Fit, FitPredict, Predict):
    def __init__(
        self,
        max_depth=None,
        max_leaves=None,
        max_bin=None,
        grow_policy=None,
        learning_rate=None,
        n_estimators=100,
        verbosity=None,
        objective=None,
        booster=None,
        tree_method=None,
        n_jobs=None,
        gamma=None,
        min_child_weight=None,
        max_delta_step=None,
        subsample=None,
        sampling_method=None,
        colsample_bytree=None,
        colsample_bylevel=None,
        colsample_bynode=None,
        reg_alpha=None,
        reg_lambda=None,
        scale_pos_weight=None,
        base_score=None,
        random_state=None,
        num_parallel_tree=None,
        monotone_constraints=None,
        interaction_constraints=None,
        importance_type=None,
        gpu_id=None,
        validate_parameters=None,
        predictor=None,
        enable_categorical=False,
        max_cat_to_onehot=None,
        eval_metric=None,
        early_stopping_rounds=None,
        callbacks=None,
        **kwargs
    ):
        self.fname = kwargs["fname"]
        self.__xgb_mcpu = xgb.dask.DaskXGBRegressor()
        return self.__xgb_mcpu.load_model(self.fname)

    def _lazy_predict_cpu(self, X, sample_weight=None, **kwargs):
        return self.__xgb_mcpu.predict(X=X, **kwargs)

    def _predict_cpu(self, X, sample_weight=None, **kwargs):
        return self.__xgb_mcpu.predict(X=X, **kwargs)


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


def create_pipeline(ml_model, dataset_path: str, x: int, y: int, z: int, executor: DaskPipelineExecutor, pipeline_save_location: str = None) -> Tuple[Pipeline, Callable]:
    # Cria o pipeline DASF para ser executado
    print("Criando pipeline....")

    # Carregamos o modelo
    xgboost = XGBRegressor(fname=ml_model)

    # Declarando os operadores necessários
    dataset   = MyDataset(name="F3 dataset", data_path=dataset_path)
    arrays2df = ArraysToDataFrame()
    persist   = PersistDaskData()

    # Convertendo valores dos vizinhos
    x = int(x)
    y = int(y)
    z = int(z)
    
    # Compondo o pipeline
    pipeline = Pipeline(
        name = f"F3 seismic",
        executor = executor)

    pipeline.add(dataset)

    # Adiciona os vizinhos ao dataset
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
        pipeline.add(arrays2df, dataset=dataset, **dict_n)

    else:
        pipeline.add(arrays2df, dataset=dataset)
    
    pipeline.add(xgboost.predict, X=arrays2df)
    
    try:
        pipeline_save_location = f"pipelines/run_n_{2*x + 2*y +2*z}"
        pipeline.visualize(filename=pipeline_save_location)

    except:
        print("error visualize")

    # Retorna o pipeline e o operador kmeans, donde os resultados serão obtidos
    return pipeline, xgboost.predict


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
    parser.add_argument("--ml-model",       type=str, required=True, help="Nome do modelo treinado .json")
    parser.add_argument("--data",           type=str, required=True, help="Caminho para o arquivo .zarr")
    parser.add_argument("--samples-window", type=str, default=0,     help="Número de vizinhos na dimensão das amostras de um traço")
    parser.add_argument("--trace-window",   type=str, default=0,     help="Número de vizinhos na dimensão dos traços de uma inline")
    parser.add_argument("--inline-window",  type=str, default=0,     help="Número de vizinhos na dimensão das inlines")
    parser.add_argument("--address",        type=str, default=None,  help="Endereço do dask scheduler. Formato: HOST:PORT")
    parser.add_argument("--output",         type=str, required=True, help="Nome do arquivo de saída onde deve ser gravado o atributo sísmico produzido")
    args = parser.parse_args()

    client = Client(args.address.replace("tcp://", ""))
    with performance_report(filename=f"reports/run_1_workers_{args.samples_window}_{args.trace_window}_{args.inline_window}.html"):
   
        # Criamos o executor
        executor = create_executor(args.address)

        # Depois o pipeline
        pipeline, last_node = create_pipeline(args.ml_model, args.data, args.samples_window, args.trace_window, args.inline_window, executor, pipeline_save_location=args.output)

        # Executamos e pegamos o resultado
        res = run(pipeline, last_node)
        print(f"O resultado é um array com o shape: {res.shape}")

        # Podemos fazer o reshape e printar a primeira inline
        inline_name = f"inlines/CIP_inline_{args.samples_window}_{args.trace_window}_{args.inline_window}.jpeg"
        res = res.values.reshape((401, 701, 255))
        plt.imsave(inline_name, res[0], cmap="viridis")
        print(f"Figura da inline 0 salva em {inline_name}")
    
