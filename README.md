# Unicamp-MO810

Execution single node
```
docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 train_model.py --data data/F3_train.zarr --attribute COS-INST-PHASE --samples-window 1 --trace-window 1 --inline-window 1 --output model.json 
```

Clear docker to restart
```
docker system prune -a
```
