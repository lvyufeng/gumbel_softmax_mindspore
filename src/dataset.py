import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV

def create_dataset(data_path, mode, batch_size=32, shuffle=True, num_parallel_workers=1, drop_remainder=False):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path, mode)

    # define map operations
    hwc2chw_op = CV.HWC2CHW()
    rescale_nml_op = CV.Rescale(1.0 / 255.0, 0)

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    if shuffle:
        mnist_ds = mnist_ds.shuffle(buffer_size=1024)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=drop_remainder)

    return mnist_ds

if __name__ == '__main__':
    path = './dataset'
    train_ds = create_dataset(path, 'train')
    assert train_ds.get_dataset_size() == 1875
    for i in train_ds.create_tuple_iterator():
        print(i)
        break
