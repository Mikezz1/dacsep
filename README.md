# Target Speaker Extraction with Neural Audio Codecs

```
pip install --target /mike_migrate2/packages4 --upgrade  -r requirements.txt
```

```
export PYTHONPATH="/mike_migrate2/packages4:$PYTHONPATH"
```

```
rsync \
    --recursive \
    --links \
    --perms \
    --compress \
    --times \
    --verbose \
    --progress \
    "mike-ix4-devbox.stt.mlc:/mike_migrate2/codec-source-sep"\
    "/Users/m.v.oleynik/Personal/codec-source-sep"

```

```
/ml_core_binaries/devbox-agent proc spawn -- python train.py
```


```
python /mike_migrate2/packages4/tensorboard/main.py --logdir <your_log_dir>
```