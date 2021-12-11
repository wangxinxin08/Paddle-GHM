import paddle


# do transpose on a list of tensors
def batch_transpose(tensors, perm):
    trans = []
    for tensor in tensors:
        trans.append(tensor.transpose(perm))
    return trans

# do reshape on a list of tensors
def batch_reshape(tensors, shape):
    reshape = []
    for tensor in tensors:
        reshape.append(tensor.reshape(shape))
    return reshape

# turn labels to one-hot vectors
def expand_onehot(labels, label_channels):
    # paddle.full does not support place as argument
    expand = paddle.to_tensor(
        paddle.full((labels.shape[0], label_channels+1), 0, dtype=paddle.float32), place=labels.place)
    expand[paddle.arange(labels.shape[0]), labels] = 1
    return expand[:, :-1]

def zero_loss(like):
    return paddle.to_tensor([0], dtype=like.dtype, place=like.place, stop_gradient=False)
