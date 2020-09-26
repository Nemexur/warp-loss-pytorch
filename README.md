# WARPLoss

Implementation of [WARP Loss](https://static.googleusercontent.com/media/research.google.com/ru//pubs/archive/37180.pdf) for MultiLabel target in PyTorch. It also supports Binary and MultiClass if you rewrite them as MultiLabel classification.

This implementation has only one for loop over batches as I wanted to make a single model that works with both MultiLabel and binary tasks.

## How to use

```python
from warp_loss import WARPLoss


# Set max number of trials to 100
loss_func = WARPLoss(max_num_trials=100)
# Pass logits as inputs and target
loss = loss_func(inputs, target)
loss.backward()
```

## What's next

1. Rewrite it without for loop and make it fully compatible with GPU parallelism seems to be pretty possible for Binary and MultiClass tasks so I would try it sometime.
2. Implement current `WARPLoss` model on `CUDA C++` to efficiently work with for loop over batch size.
3. Implement some kind of masking for operations with positives and negatives in `WARPLoss`. Maybe it would eliminate the need for custom backward and for loop over batch size.
