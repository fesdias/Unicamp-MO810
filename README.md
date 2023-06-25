# Unicamp-MO810

Execution single node
```
docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 train-model.py --data data/F3_train.zarr --attribute COS-INST-PHASE --samples-window 1 --trace-window 1 --inline-window 1 --output model.json 
```
Execution parallel x, y, z = 4
```
docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 train-model.py --data data/F3_train.zarr --attribute COS-INST-PHASE --samples-window 4 --trace-window 4 --inline-window 4 --output model.json --address

```
Clear docker to restart
```
docker system prune -a
```
```
docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 train-model.py --attribute CIF --data data/F3_train.zarr --inline-window 2 --trace-window 0 --samples-window 0 --address tcp://143.106.16.207:8786 --output models/CIF-ml-model-2-0-0.json
```
