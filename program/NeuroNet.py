import scipy.special
import numpy as np

def log(*a):
    # print(*a)
    pass

class Layer():

    def __init__(self, name, size):
        self.name = name
        self.size = size       # number of neurons is this layer
        self.w = None          # weight matrix
        self.b = None          # bias vector
        self.previous = None   # reference to layer below (if any)
        self.excitation = None # vector
        self.error = None      # vector

    def set(self, _pattern):
        self.excitation = _pattern

    def get_excitation(self):
        return self.excitation

    def get_error(self):
        ret_val = np.dot(self.w.transpose(), self.error)
        return ret_val

    def forward(self):
        _sum = np.dot(self.w, self.previous.get_excitation()) + self.b
        self.excitation = scipy.special.expit(_sum)

    def connect(self, previous):
        self.previous = previous
        self.w = np.random.normal(0.0, pow(previous.size, -0.5), (self.size, previous.size))
        self.b = np.random.normal(0.0, pow(previous.size, -0.5), self.size)

    def backward(self, err, alpha):
        self.error = err
        # Calculate adjustment factors
        val_tmp = self.excitation * (1 - self.excitation)
        log(self.name, ": val_tmp=", val_tmp, type(val_tmp))
        val_next = alpha * err * val_tmp
        log(self.name, ": val_next=", val_next, type(val_next))

        # adjust bias
        self.b += val_next
        log(self.name, ": b =", self.b)

        # adjust weights
        delta_w = np.outer(val_next, self.previous.get_excitation())
        log(type(delta_w), self.name, ": delta_w=")
        log(delta_w)
        self.w += delta_w
        log(self.name, ": w =", self.w)

    def to_dict(self):
        data = {'weights': self.w.tolist(), 'bias': self.b.tolist()}
        return data

class NeuroNet():

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, random_seed):
        log("NeuroNet.initialize: input_layer_size =", input_layer_size,
            ", hidden_layer_size =", hidden_layer_size,
            ", output_layer_size =", output_layer_size,
            ", random_seed =", random_seed)

        np.random.seed(random_seed)
        self.input_layer = Layer("input layer", input_layer_size)
        self.hidden_layer = Layer("hidden layer", hidden_layer_size)
        self.output_layer = Layer("output layer", output_layer_size)
        if hidden_layer_size > 0:
            self.hidden_layer.connect(self.input_layer)
            self.output_layer.connect(self.hidden_layer)
        else:
            self.output_layer.connect(self.input_layer)

    def train(self, all_pattern, alpha):
        num_pattern = len(all_pattern)
        log("NeuroNet.train: number of pattern =", num_pattern,
            ", alpha =", alpha)

        error_epoch = 0
        for i in range(num_pattern):
            # Select and apply training pattern
            output_index, data = all_pattern[i] # output_index = expected label
            self.input_layer.set(data)

            # Forward propagation
            if self.hidden_layer.size > 0:
                self.hidden_layer.forward()
            self.output_layer.forward()

            # Calculate error
            exp = np.zeros(self.output_layer.size)
            exp[int(output_index)] = 1
            error = exp - self.output_layer.get_excitation()
            
            log("error", error, type(error))
            error_total = (error ** 2).sum()
            log("error_total =", error_total)
            error_epoch += error_total

            # Perform back propagation
            self.output_layer.backward(error, alpha)
            if self.hidden_layer.size > 0:
                error = self.output_layer.get_error()
                self.hidden_layer.backward(error, alpha)

        return error_epoch / num_pattern

    def run(self, pattern_data):
        self.input_layer.set(pattern_data)
        if self.hidden_layer.size > 0:
            self.hidden_layer.forward()
            hidden_layer_excitation = self.hidden_layer.get_excitation()
        else:
            hidden_layer_excitation = None
        self.output_layer.forward()
        return hidden_layer_excitation, self.output_layer.get_excitation()

    def dump(self):
        print('Dumping trained network')
        if self.hidden_layer.size > 0:
            return {'hidden_layer': self.hidden_layer.to_dict(),
                    'output_layer': self.output_layer.to_dict()}
        else:
            return {'output_layer': self.output_layer.to_dict()}