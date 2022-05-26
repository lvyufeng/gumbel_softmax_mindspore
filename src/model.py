import math
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import ms_function
from mindspore.common.initializer import initializer, HeUniform, Uniform, _calculate_fan_in_and_fan_out

def gumbel_softmax(logits, temperature, hard, axis=-1, eps=1e-20):
    uniform_samples = ops.UniformReal()(logits.shape)
    gumbels = -ops.log(-ops.log(uniform_samples + eps) + eps) # ~Gumbel(0, 1)
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

class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=has_bias, activation=activation)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))

class VAE_Gumbel(nn.Cell):
    def __init__(self, latent_dim, class_num):
        super().__init__()
        self.latent_dim = latent_dim
        self.class_num = class_num
        self.fc1 = Dense(784, 512, activation='relu')
        self.fc2 = Dense(512, 256, activation='relu')
        self.fc3 = Dense(256, latent_dim * class_num, activation='relu')

        self.fc4 = Dense(latent_dim * class_num, 256, activation='relu')
        self.fc5 = Dense(256, 512, activation='relu')
        self.fc6 = Dense(512, 784, activation='sigmoid')

    def encode(self, inputs):
        outputs = self.fc1(inputs)
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)
        return outputs
    
    @ms_function
    def decode(self, hidden):
        outputs = self.fc4(hidden)
        outputs = self.fc5(outputs)
        outputs = self.fc6(outputs)
        return outputs
        
    def construct(self, inputs, temp, hard):
        q = self.encode(inputs.view(-1, 784))
        q_y = q.view(q.shape[0], self.latent_dim, self.class_num)
        z = gumbel_softmax(q_y, temp, hard)
        z = z.view(-1, self.latent_dim * self.class_num)
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
        return loss, ops.stop_gradient(recon_batch)

class TrainOneStep(nn.TrainOneStepCell):
    def __init__(self, model, optim):
        super().__init__(model, optim)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data, temp, hard):
        loss, recon_batch = self.network(data, temp, hard)
        grads = self.grad(self.network, self.weights)(data, temp, hard)
        self.optimizer(grads)
        return loss, recon_batch