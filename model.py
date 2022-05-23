import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


def gumbel_softmax(logits, temperature, hard, axis=-1):
    uniform_samples = ops.UniformReal()(logits.shape)
    gumbels = -ops.log(-ops.log(uniform_samples)) # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / temperature
    y_soft = ops.Softmax(axis)(gumbels)

    if hard:
        # Straight through
        index = y_soft.argmax(axis)
        y_hard = ops.OneHot(axis)(index, y_soft.shape[axis], ops.scalar_to_array(1.0), ops.scalar_to_array(0.0))
        ret = ops.stop_gradient(y_hard - y_soft) + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class VAE_Gumbel(nn.Cell):
    def __init__(self, latent_dim, class_num):
        super().__init__()
        self.latent_dim = latent_dim
        self.class_num = class_num
        self.fc1 = nn.Dense(784, 512, activation='relu')
        self.fc2 = nn.Dense(512, 256, activation='relu')
        self.fc3 = nn.Dense(256, latent_dim * class_num, activation='relu')

        self.fc4 = nn.Dense(latent_dim * class_num, 256, activation='relu')
        self.fc5 = nn.Dense(256, 512, activation='relu')
        self.fc6 = nn.Dense(512, 768, activation='sigmoid')
    
    def encode(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)
        return outputs
    
    def decode(self, hidden):
        outputs = self.fc4(hidden)
        outputs = self.fc5(hidden)
        outputs = self.fc6(hidden)
        return outputs
        
    def construct(self, inputs, temp, hard):
        q = self.encode(inputs.view(-1, 784))
        q_y = q.view(q.shape[0], self.latent_dim, self.class_num)
        z = gumbel_softmax(q_y, temp, hard)
        return self.decode(z), ops.Softmax()(q_y).reshape(q.shape)

class Loss(nn.Cell):
    """Reconstruction + KLDiv Kisses summed over all elements and batch"""
    def __init__(self, class_num):
        super().__init__()
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.class_num = class_num
        self.eps = 1e-20

    def construct(self, recon_x, x, qy):
        bce = self.bce_loss(recon_x, x.view(-1, 784)) / x.shape[0]
        log_ratio = ops.log(qy * self.class_num + self.eps)
        kl_div = ops.reduce_sum(qy * log_ratio, -1).mean()
        return bce + kl_div

class NetWithLoss(nn.Cell):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss

    def construct(self, data, temp, hard):
        recon_batch, qy = self.model(data, temp, hard)
        loss = self.loss(recon_batch, data, qy)
        return loss

class TrainOneStep(nn.TrainOneStepCell):
    def __init__(self, model, optim):
        super().__init__(model, optim)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data, temp, hard):
        loss = self.model(data, temp, hard)
        grads = self.grad(self.model, self.weights)(data, temp, hard)
        self.optimizer(grads)
        return loss

if __name__ == '__main__':
    pass