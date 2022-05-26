import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.numpy as mnp
from tqdm import tqdm
from src.model import VAE_Gumbel, Loss, NetWithLoss, TrainOneStep
from src.dataset import create_dataset
from src.utils import to_image

temp_min = 0.5
ANNEAL_RATE = 0.00003
latent_dim = 30
class_num = 10  # one-of-K vector

def arg_parse():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--data_path', type=str, default='./dataset',
                        help='dataset path (default: ./dataset)')
    parser.add_argument('--device', type=str, default='GPU',
                        help='device target (default: GPU)')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--temp', type=float, default=1.0, metavar='S',
                        help='tau(temperature) (default: 1.0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--hard', action='store_true', default=False,
                        help='hard Gumbel softmax')
    args = parser.parse_args()
    return args

def train(model, dataset, epoch, args):
    model.set_train()
    total = dataset.get_dataset_size()
    loss_total = 0
    step_total = 0
    temp = args.temp

    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for (data, _) in dataset.create_tuple_iterator():
            loss, _ = model(data, temp, args.hard)
            loss_total += loss.asnumpy()
            step_total += 1

            if step_total % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * step_total), temp_min)
            t.set_postfix(loss='{:.6f}'.format(loss_total/step_total))
            t.update(1)

def test(model, dataset, epoch, args):
    model.set_train(False)
    total = dataset.get_dataset_size()
    epoch_loss = 0
    step_total = 0
    temp = args.temp

    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for (data, _) in dataset.create_tuple_iterator():
            loss, recon_batch = model(data, temp, args.hard)
            epoch_loss += loss.asnumpy()

            if step_total % 100 == 1:
                temp = np.maximum(temp * np.exp(-ANNEAL_RATE * step_total), temp_min)
            if step_total == 0:
                n = min(data.shape[0], 8)
                comparison = mnp.concatenate([data[:n],
                                             recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                to_image(comparison, args.data_path + '/reconstructions/reconstruction_' + str(epoch) + '.png', nrow=n)

            step_total += 1
            t.set_postfix(loss='{:.6f}'.format(epoch_loss/step_total))
            t.update(1)

    return epoch_loss / total

def sample(model, epoch):
    M = 64 * latent_dim
    np_y = np.zeros((M, class_num), dtype=np.float32)
    np_y[range(M), np.random.choice(class_num, M)] = 1
    np_y = np.reshape(np_y, [M // latent_dim, latent_dim, class_num])
    sample = mindspore.Tensor(np_y).view(M // latent_dim, latent_dim * class_num)
    sample = model.decode(sample)
    to_image(sample.view(M // latent_dim, 1, 28, 28),
             args.data_path + '/samples/sample_' + str(epoch) + '.png')

def run(args):
    mindspore.set_seed(args.seed)
    # mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

    net = VAE_Gumbel(latent_dim, class_num)
    net_with_loss = NetWithLoss(net, Loss(class_num))
    optim = nn.Adam(net.trainable_params(), learning_rate=1e-3)
    trainer = TrainOneStep(net_with_loss, optim)

    train_dataset = create_dataset(args.data_path, 'train', args.batch_size, drop_remainder=True)
    test_dataset = create_dataset(args.data_path, 'test', args.batch_size, drop_remainder=True)


    for epoch in range(1, args.epochs + 1):
        train(trainer, train_dataset, epoch, args)
        test(net_with_loss, test_dataset, epoch, args)
        sample(net, epoch)


if __name__ == '__main__':
    args = arg_parse()
    run(args)
