
import warnings

import numpy as np
import os
from tensorflow.keras.models import load_model

from snntoolbox.parsing.utils import get_type
from snntoolbox.simulation.utils import AbstractSNN, get_shape_from_label, \
    build_convolution, build_pooling, get_ann_ops
from snntoolbox.utils.utils import confirm_overwrite


class SNN(AbstractSNN):
    """

    Attributes
    ----------

    layers: list[spaic.NeuronGroup]

    connections: list[spaic.Connection]

    v_th: float32

    neuron_model:

    spikemonitors:

    statemonitors:

    snn: spaic.Network

    """

    def __init__(self, config, queue=None):

        AbstractSNN.__init__(self, config, queue)
        self.layers = []
        self.connection = []
        # TODO: 怎么设置每层的v_th?
        self.v_th = 10
        self.neuron_model = 'IF'
        self.statemonitors = []
        self.spikemonitors = []
        self.snn = None
        self._input_layer = None

        self.output_spikemonitor = None

    @property
    def is_parallelizable(self):
        return False

    def add_input_layer(self, input_shape):

        self.layers.append(self.sim.NeuronGroup(
            neuron_number=np.prod(input_shape[1:]),
            neuron_model=self.neuron_model, v_th=self.v_th))
        self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[0]))
        self.statemonitors.append(self.sim.StateMonitor(self.layers[0], 'V'))


    def add_layer(self, layer):

        # Latest Keras versions need special permutation after Flatten layers.
        if 'Flatten' in layer.__class__.__name__ and \
                self.config.get('input', 'model_lib') == 'keras':
            self.flatten_shapes.append(
                (layer.name, get_shape_from_label(self.layers[-1].label)))
            return

        self.layers.append(self.sim.NeuronGroup(
            neuron_number=np.prod(layer.output_shape[1:]),
            neuron_model=self.neuron_model, v_th=self.v_th))
        # spaic创建spaic.Connection需要立即给出权重，所以此处不创建连接

        # 不考虑AbstractSNN的细节，暂时直接创建SpikeMonitor和StateMonitor
        self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[-1]))
        self.statemonitors.append(self.sim.StateMonitor(self.layers[-1], 'V'))


    def build_dense(self, layer, weights=None):

        if layer.activation == 'softmax':
            raise warnings.warn("Activation 'softmax' not implemented. Using "
                                "'relu' activation instead.", RuntimeWarning)

        _weights, biases = layer.get_weights()
        if weights is None:
            weights = _weights

        self.set_biases(biases)

        # TODO: spaic神经元模型暂时没用到delay
        delay = self.config.getfloat('cell', 'delay')

        if len(self.flatten_shapes) == 1:
            print("Swapping data_format of Flatten layer.")
            flatten_name, shape = self.flatten_shapes.pop()

        # TODO: 暂时不处理Flatten
        if self.data_format == 'channels_last':
            pass
        elif len(self.flatten_shapes) > 1:
            raise RuntimeWarning("Not all Flatten layers have been consumed.")
        else:
            # 转换为spaic权重格式：权重weight[i][j]对应从net.input第j个神经元到net.output第i个神经元的连接。
            weights = np.array(weights).transpose()
            self.connection.append(self.sim.Connection(
                self.layers[-2], self.layers[-1],
                link_type='full', weight=weights))


    def build_convolution(self, layer, weights=None):

        delay = self.config.getfloat('cell', 'delay')
        transpose_kernel = \
            self.config.get('simulation', 'keras_backend') == 'tensorflow'
        conns, biases = build_convolution(layer, delay, transpose_kernel)
        connections = np.array(conns)

        # TODO: 验证spaic.NeuronGroup神经元数量是否为 spaic.NeuronGroup.num
        weights = np.zeros((self.layers[-2].num, self.layers[-1].num))
        for i, j, w, _ in connections:
            weights[j][i] = w

        self.set_biases(biases)

        print("Connecting layer...")

        # TODO: spaic只支持全连接，卷积转换后一定是全连接吗？将connections中不存在的神经元连接weight设为0

        self.connection.append(self.sim.Connection(
            self.layers[-2], self.layers[-1],
            link_type='full', weight=weights))


    def build_pooling(self, layer, weights=None):

        delay = self.config.getfloat('cell', 'delay')
        connections = np.array(build_pooling(layer, delay))

        # TODO: 验证spaic.NeuronGroup神经元数量是否为 spaic.NeuronGroup.num
        weights = np.zeros((self.layers[-2].num, self.layers[-1].num))
        for i, j, w, _ in connections:
            weights[j][i] = w

        print("Connecting layer...")

        # TODO: spaic只支持全连接，卷积转换后一定是全连接吗？将connections中不存在的神经元连接weight设为0
        self.connection.append(self.sim.Connection(
            self.layers[-2], self.layers[-1],
            link_type='full', weight=weights))


    def compile(self):

        self.output_spikemonitor = self.sim.SpikeMonitor(self.layers[-1])
        self.snn = self.sim.Network(self.config.get('path', 'filename_ann'))

        self.snn.input = self.layers[0]

        for id, layer in enumerate(self.layers[1:]):
            self.snn.add_assembly(name='layer'+str(id), assembly=layer)

        for id, conn in enumerate(self.connection):
            self.snn.add_connection(name='connection'+str(id), connection=conn)

        # TODO: 添加所有层spikemonitor与statemonitor。目前只添加输出层spikemonitor
        self.snn.output_spikemonitor = self.sim.SpikeMonitor(self.layers[-1])


    def simulate(self, **kwargs):

        time_window = self.config.getint('simulation', 'duration')
        dt = self.config.getfloat('simulation', 'dt')

        inputs = kwargs[str('x_b_l')]
        spikes = self.encode(inputs, time_window, dt)

        sim = self.sim.Backend(self.snn, 'emu', dt=0.1, device={'platform':'simulator'})

        self.snn.input(spikes)
        sim.run(time_window)

        # TODO: self.get_recorded_vars() 调用了 self.get_spiketrains(), self.get_spiketrains_input()
        #  这两个函数没有重载：缺少input和中间层的spikemonitor。似乎与plot有关。
        output_b_l_t = self.get_recorded_vars(self.layers)

        return output_b_l_t


    def set_biases(self, biases):
        pass

    def encode(self, input, time_window=20, dt=0.1):
        '''
        编码输入数据

        Parameters
        ----------
        testdata
        time_window
        dt

        Returns
        -------
        spikes: array[batch, time, neuron]

        '''
        data_type = self.config.get('simulation', 'data_type')

        # TODO: 默认灰度图像，数值范围0-255，1 channel
        if data_type == 'vision':
            batch_size = input.shape[0]
            flatten_img = input.reshape(batch_size, -1)/255
            spikes = np.array([np.random.rand(
                int(time_window / dt), flt_img.size).__le__(flt_img * dt).astype(float) for flt_img in flatten_img])
            return spikes
        # TODO: 音频信号编码
        elif data_type == 'audio':
            raise NotImplementedError


    def get_spiketrains_output(self):

        shape = [self.batch_size, self.num_classes, self._num_timesteps]

        # spaic的输出脉冲格式为 [batch, neuron_id, time]
        spiketrain_b_t_l = self.snn.output_spikemonitor.spikes
        spiketrain_b_l_t = self.reshape_spiketrain(spiketrain_b_t_l)

        return spiketrain_b_l_t


    def reshape_spiketrain(self, spiketrain_b_t_l):
        return np.array([k.transpose() for k in spiketrain_b_t_l])