tensorboard command:
```python -m tensorflow.tensorboard --logdir=[log directory]```
floyd command:
```floyd run --gpu --data [data id] "python somefile.py"```
floyd의 output은 /output/에 저장해야 인식됨.