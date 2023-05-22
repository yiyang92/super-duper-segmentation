### Semantic segmentation

Change datasets paths in params.py, use parameters overloading.
To train:
```python
train_segmenter -p u_net_baseline
```

# TODO

This project is a prorotype for semantic segmentation. For better performance:

- Need data augmentation (e.g. similar to Unet paper). Model is suffering from heavy overfitting.
- Use pretrained encoders for Unet - e.g. Resnet for better starting representation.
- Use transformer-like models from segment anything paper, seems Sota for now
- Add more data (but yeah maybe not a good choice so use pre-trained encoder).
- Add multi-gpu support.
