from .type import *
from .model import *
from .devices import *
from .config import *
RAMPATH = "./ramulator2"
RAMLOG = "./ramulator.out"

OPB_PRINT = False


class System:

    def __init__(self,
                 gpu_config,
                 modelinfos=None,
                 hetero_name: DeviceType = DeviceType.NONE,
                 hetero_config=None):
        scaling_factor = SCALING_FACTOR
        self.hetero_name = hetero_name
        self.GPU = xPU(DeviceType.GPU, gpu_config, scaling_factor)
        self.AttDevice = self.GPU
        if self.hetero_name == DeviceType.PIM:
            self.AttDevice = PIM(hetero_config, scaling_factor)

        elif self.hetero_name == DeviceType.CPU:
            self.AttDevice = xPU(DeviceType.CPU, hetero_config, scaling_factor)

        self.devices = {'GPU': self.GPU, 'Acc': self.AttDevice}

        self.model_set = 0
        if modelinfos is not None:
            self.model = Transformer(modelinfos,
                                     tensor_parallel=self.GPU.num_xpu) # 根据num_xpu做张量并行
            self.model_set = 1

        self.scaling_factor = scaling_factor

    def set_model(self, modelinfos):
        self.model = Transformer(modelinfos, tensor_parallel=self.GPU.num_xpu)
        self.model_set = 1

    def set_accelerator(self, modelinfos, name: DeviceType, config):
        self.hetero_name = name
        if self.hetero_name == DeviceType.PIM:
            ramulator = Ramulator(modelinfos, "ramulator2", "ramulator.out")
            self.devices['Acc'] = PIM(config,
                                      self.scaling_factor,
                                      ramulator)

        elif self.hetero_name == DeviceType.CPU:
            self.devices['Acc'] = xPU(DeviceType.CPU, config,
                                      self.scaling_factor)

    # Set all device to GPU
    def set_xpu(self, config):
        self.hetero_name = DeviceType.NONE
        self.GPU = xPU(DeviceType.GPU, config, self.scaling_factor)
        self.devices['GPU'] = self.GPU
        self.devices['Acc'] = self.GPU
        self.model.tp = self.GPU.num_xpu

    def simulate(self,
                 batch_size,
                 lin,
                 lout,
                 perfs=None,
                 pipe=False,
                 parallel_ff=False,
                 power_constraint=False,
                 num_reqs=0):

        def add_infos(name, infos, time, energy, bound):
            new_name = name
            if new_name in infos.keys():
                infos[new_name]["time"] += time
                infos[new_name]["energy"] = [
                    eng + energy[i]
                    for i, eng in enumerate(infos[new_name]["energy"])
                ]
            else:
                infos[new_name] = {
                    "time": time,
                    "energy": energy,
                    "bound": bound
                }

        def acc_time(type, exec_times, exec_time):
            if type in exec_times.keys():
                exec_times[type] += exec_time
            else:
                exec_times[type] = exec_time

        def acc_energy(type, energies, energy):
            if type in energies.keys():
                energy_ = energies[type]
                energies[type] = [
                    energy_[i] + energy[i] for i in range(len(energy_))
                ]
            else:
                energies[type] = energy

        def _opb_print(layer, stage_name):
            if OPB_PRINT and layer.off_traffic != 0:
                opb = layer.get_flops() / layer.off_traffic
                tflops = layer.get_flops(
                ) / exec_time / 1000 / 1000 / 1000 / 1000
                print("{},{},{},{},{},{}".format(stage_name, batch_size, lin,
                                                 layer.name, opb, tflops))

        def _pipeline(layers, level=False):
            qkv_time, prj_time, score_time, context_time, x2g_time, softmax_time = 0, 0, 0, 0, 0, 0
            for layer in layers:
                if layer.name in ["qkv"]:
                    qkv_time += layer.exec_time
                elif layer.name in ["proj"]:
                    prj_time += layer.exec_time
                elif layer.name in ["comm_x2g"]:
                    x2g_time += layer.exec_time
                elif layer.name in ["score"]:
                    score_time += layer.exec_time
                elif layer.name in ["context"]:
                    context_time += layer.exec_time
                elif layer.name in ["softmax"]:
                    softmax_time += layer.exec_time
            # 单卡？
            minimum_ratio = 1 / (self.model.num_heads / self.GPU.num_xpu) #一个GPU能分几个头
            if level == False: # 仅考虑 Attention 内部流水的简化模型。
                #softmax_time = 0
                attn_time = score_time + context_time + softmax_time
                if attn_time > x2g_time: 
                    # 计算能够掩盖通信流水，只需要计算第一个 head 的流水启动通信，加上最后一个 head 的流水排空通信
                    # x2g_time的单位是所有 head 的流水时间单元是一次GPU->PIM->GPU, 前半段是KV传输，后半段是context传输
                    x2g_time *= minimum_ratio
                else:
                    x2g_time -= attn_time * (1 - minimum_ratio)

            else:
                #softmax_time = 0
                fc_time = qkv_time + prj_time
                attn_time = score_time + context_time + softmax_time
                if attn_time > fc_time:
                    qkv_time *= minimum_ratio
                    prj_time *= minimum_ratio

                    if attn_time > x2g_time:
                        x2g_time *= minimum_ratio
                    else:
                        x2g_time -= attn_time * (1 - minimum_ratio)
                else:
                    if fc_time > x2g_time:
                        x2g_time *= minimum_ratio
                        qkv_time -= attn_time * (1 - minimum_ratio) * (3 / 4)
                        prj_time -= attn_time * (1 - minimum_ratio) * (1 / 4)
                    else:
                        x2g_time -= attn_time * (1 - minimum_ratio)
                        qkv_time *= minimum_ratio
                        prj_time *= minimum_ratio
            softmax_time = 0

            for layer in layers:
                if layer.name in ["qkv"]:
                    layer.exec_time = qkv_time
                elif layer.name in ["proj"]:
                    layer.exec_time = prj_time
                elif layer.name in ["comm_x2g"]:
                    # for 2 comm_x2g layers
                    # decoder 是一个 Python 列表，构造时向里面放入了两个不同的 Layer 对象，但它们的名字相同。
                    layer.exec_time = x2g_time / 2
                elif layer.name in ["softmax"]:
                    layer.exec_time = softmax_time

        def _ff_parallel(layers):
            # 让 GPU 和 PIM 一起算 FFN
            bw_scale = self.devices['Acc'].peak_memory_bandwidth / self.devices[
                'GPU'].peak_memory_bandwidth
            for layer in layers:
                if "ff" in layer.name:
                    if layer.bound == "compute": # 计算受限
                        attn_flops = self.devices[
                            'GPU'].peak_memory_bandwidth / layer.dbyte * 2 * bw_scale
                        ratio = self.devices['GPU'].peak_flops / (
                            self.devices['GPU'].peak_flops + attn_flops)
                        layer.exec_time *= ratio

                    elif layer.bound == "memory": # 访存受限
                        attn_eff_bw = self.devices[
                            'GPU'].peak_memory_bandwidth * bw_scale / bs
                        ratio = self.devices['GPU'].peak_memory_bandwidth / (
                            self.devices['GPU'].peak_memory_bandwidth +
                            attn_eff_bw)
                        layer.exec_time *= ratio

        assert self.model_set, "Need to set_model"
        self.model.build(batch_size, lin, lout, self.hetero_name
                         in [DeviceType.CPU, DeviceType.PIM])
        second_batch_size = num_reqs % batch_size
        num_batches = 1
        target_bs = [batch_size]
        if num_reqs > 0:
            num_batches = int(num_reqs / batch_size)
            if second_batch_size > 0:
                target_bs = [batch_size, second_batch_size]

        s_flops = 0
        g_flops = 0

        gen_energies = {} #gen阶段按算子(layer)的划分统计

        unit_energy = {
            'g_all': 0,
            'g_offmem': 0,
            'g_l2': 0,
            'g_l1': 0,
            'g_reg': 0,
            'g_alu': 0,
            'g_comm': 0
        }

        perf_all = []
        energy_all = []
        for itr, bs in enumerate(target_bs):
            time = 0
            wrt_io_busy = 0 # 上一次写结束的时间戳，表示写传输通道要忙到什么时刻才空闲
            s_decoder = self.model.sum_decoder
            g_decoder = self.model.gen_decoder

            ## Summarization stage
            for layer in s_decoder:
                # Get execution time and energy
                exec_time, energy = self.devices['GPU'].get_time_and_energy(
                    layer)

                # Time to transfer KV matrices to memory (PCIe bandwidth)
                if layer.type == LayerType.X2G:
                    exec_time += max(wrt_io_busy - time, 0) # 可能会等待
                    wrt_io_busy = time + exec_time # 更新忙时时刻
                layer.exec_time = exec_time
                layer.energy = energy

                s_flops += layer.get_flops() * self.devices['GPU'].num_xpu
                time += exec_time
                _opb_print(layer, 'sum')

            ## Generation stage
            for gen_stage, decoder_block in enumerate(g_decoder): # 这里分了每个decode iteration
                for l_idx, layer in enumerate(decoder_block):
                    # Get execution time and energy
                    if layer.type in [
                            LayerType.MATMUL, LayerType.SOFTMAX, LayerType.X2G
                    ]:
                        exec_time, energy = self.devices[
                            'Acc'].get_time_and_energy(layer)
                    else:
                        exec_time, energy = self.devices[
                            'GPU'].get_time_and_energy(layer)
                    layer.exec_time = exec_time
                    layer.energy = energy
                    g_flops += layer.get_flops() * self.devices['GPU'].num_xpu
                    time += exec_time
                    if gen_stage == 0:
                        _opb_print(layer, 'gen')

                    # energy 
                    if layer.type in gen_energies:
                        gen_energies[layer.type]['mem'] += layer.energy[0] # GPU片外HBM能耗，PIM为DRAM Cell + HBM内部数据通路能耗
                        gen_energies[layer.type]['comp'] += sum( # comp = 片上缓存与寄存器能耗 + ALU能耗
                            layer.energy[1:5])
                        gen_energies[layer.type]['comm'] += layer.energy[5] # GPU之间的 AllReduce, GPU与AttAcc之间的数据传输
                    else:
                        gen_energies[layer.type] = {}
                        gen_energies[layer.type]['mem'] = layer.energy[0]
                        gen_energies[layer.type]['comp'] = sum(
                            layer.energy[1:5])
                        gen_energies[layer.type]['comm'] = layer.energy[5]

                    unit_energy['g_all'] += sum(layer.energy)
                    # energy: [HBM能耗, L2能耗, L1能耗, 寄存器能耗, ALU能耗, 通信能耗]
                    unit_energy['g_offmem'] += layer.energy[0]
                    unit_energy['g_l2'] += layer.energy[1]
                    unit_energy['g_l1'] += layer.energy[2]
                    unit_energy['g_reg'] += layer.energy[3]
                    unit_energy['g_alu'] += layer.energy[4]
                    unit_energy['g_comm'] += layer.energy[5]

                # pipeline
                if self.hetero_name == DeviceType.PIM:
                    _pipeline(decoder_block, pipe)
                    if parallel_ff:
                        _ff_parallel(decoder_block)

            ''' #######
            
            根据前面的计算，进行分类汇总

            ''' #######

            s_perf = {
                'all': 0,
                'matmul': 0,
                'fc': 0,
                'comm': 0,
                'softmax': 0,
                'act': 0,
                'norm': 0
            }
            for layer in s_decoder:
                exec_time = layer.exec_time
                if layer.type == LayerType.FC:
                    s_perf['all'] += exec_time
                    s_perf['fc'] += exec_time
                elif layer.type == LayerType.MATMUL:
                    s_perf['all'] += exec_time
                    s_perf['matmul'] += exec_time
                elif layer.type == LayerType.G2G:
                    s_perf['all'] += exec_time
                    s_perf['comm'] += exec_time
                elif layer.type == LayerType.SOFTMAX:
                    s_perf['all'] += exec_time
                    s_perf['softmax'] += exec_time
                elif layer.type == LayerType.ACT:
                    s_perf['all'] += exec_time
                    s_perf['act'] += exec_time
                elif layer.type == LayerType.NORM:
                    s_perf['all'] += exec_time
                    s_perf['norm'] += exec_time

            g_perf = {
                'all': 0,
                'matmul': 0,
                'fc': 0,
                'comm': 0,
                'etc': 0,
                'qkv': 0,
                'prj': 0,
                'ff': 0,
                'g2g': 0,
                'x2g': 0,
                'softmax': 0,
                'act': 0,
                'norm': 0
            }

            for gen_stage, decoder_block in enumerate(g_decoder):
                for l_idx, layer in enumerate(decoder_block):
                    exec_time = layer.exec_time
                    g_perf['all'] += exec_time
                    if layer.type == LayerType.FC:
                        g_perf['fc'] += exec_time
                        if 'ff' in layer.name:
                            g_perf['ff'] += exec_time
                        elif 'qkv' in layer.name:
                            g_perf['qkv'] += exec_time
                        elif 'proj' in layer.name:
                            g_perf['prj'] += exec_time
                    elif layer.type == LayerType.MATMUL:
                        g_perf['matmul'] += exec_time
                    elif layer.type in [LayerType.G2G, LayerType.X2G]:
                        g_perf['comm'] += exec_time
                        if 'x2g' in layer.name:
                            g_perf['x2g'] += exec_time
                        elif 'g2g' in layer.name:
                            g_perf['g2g'] += exec_time
                    elif layer.type in [LayerType.ACT, LayerType.NORM]:
                        g_perf['etc'] += exec_time
                        if layer.type == LayerType.ACT:
                            g_perf['act'] += exec_time
                        elif layer.type == LayerType.NORM:
                            g_perf['norm'] += exec_time
                    elif layer.type == LayerType.SOFTMAX:
                        g_perf['softmax'] += exec_time
            # 因为不同 step 的 KV cache 长度不同，所以每一步 Attention 时间也不同。这里计算平均生成一个 token 的耗时
            g_perf = {k: v / (lout - 1) for k, v in g_perf.items()}

            '''
            unit_energy   按硬件位置分类
            gen_energies 按算子类型分类
            '''
            energies = [
                unit_energy['g_all'], unit_energy['g_offmem'],
                unit_energy['g_l2'], unit_energy['g_l1'], unit_energy['g_reg'],
                unit_energy['g_alu'], gen_energies[LayerType.FC]['mem'],
                gen_energies[LayerType.FC]['comp'],
                gen_energies[LayerType.MATMUL]['mem'] +
                gen_energies[LayerType.SOFTMAX]['mem'],
                gen_energies[LayerType.MATMUL]['comp'] +
                gen_energies[LayerType.SOFTMAX]['comp'],
                gen_energies[LayerType.ACT]['mem'] +
                gen_energies[LayerType.NORM]['mem'],
                gen_energies[LayerType.ACT]['comp'] +
                gen_energies[LayerType.NORM]['comp']
            ]
            comm_energy = sum([v['comm'] for k, v in gen_energies.items()])
            energies.append(comm_energy)

            energies = [i / (lout - 1) for i in energies] # 平均每个 token、单个 decoder block 的能耗

            perf = list(s_perf.values()) + list(g_perf.values())

            cap_usage = sum(self.get_required_mem_capacity(bs, lin, lout))

            ## Scaling to all decoder
            ## Perf: ms, energy: nJ
            perf = [t * self.model.ndec * 1000 for t in perf]
            energies = [t * self.model.ndec / 1000 for t in energies]

            if itr == 0:
                if len(perf_all) > 0:
                    perf_all = [
                        v + perf[i] * num_batches
                        for i, v in enumerate(perf_all)
                    ]
                    energy_all = [
                        v + energy[i] * num_batches
                        for i, v in enumerate(energy_all)
                    ]
                else:
                    perf_all = copy.deepcopy(perf)
                    energy_all = copy.deepcopy(energies)
            else:
                perf_all = [v + perf[i] for i, v in enumerate(perf_all)]
                energy_all = [v + energy[i] for i, v in enumerate(energy_all)]

        s_flops = s_flops * self.model.ndec / (lout - 1)
        g_flops = g_flops * self.model.ndec / (lout - 1)

        ## Concat tag
        cap = self.devices['GPU'].aggregate_memory_capacity
        if self.hetero_name in [DeviceType.CPU, DeviceType.PIM]:
            cap += self.devices['Acc'].aggregate_memory_capacity
        cap = int(cap / (1024 * 1024 * 1024))
        # PIM/CPU 带宽是单张 GPU 带宽的多少倍
        bw_scale = self.devices['Acc'].peak_memory_bandwidth / self.devices[
            'GPU'].peak_memory_bandwidth

        opb = self.devices['GPU'].peak_flops / self.devices[
            'GPU'].peak_memory_bandwidth
        if self.model.dtype in [DataType.W8A8]: # INT8 时把峰值算力翻倍
            opb *= 2

        tag = [
            self.model.name, self.model.dtype.name,
            self.devices['GPU'].name.name, cap, bw_scale, opb
        ]
        config = [
            self.hetero_name.name, self.devices['GPU'].num_xpu, pipe,
            parallel_ff, power_constraint, 0, lin, lout, batch_size,
            cap_usage, s_flops, g_flops
        ]
        if self.hetero_name == DeviceType.PIM:
            config[0] = self.devices['Acc'].pim_type.name
        # 模型与硬件信息，模拟参数，时间统计，能耗统计
        output = [tag, config, perf_all, energy_all]
        print(
            "    Batch: {}, Throughput: {:.2f} tokens/s Latency: {:.2f}ms, pipe/ff_parallel: {}/{}, powerlimit: {}"
            .format(batch_size, batch_size / ((perf_all[len(s_perf)]) / 1000),
                    perf_all[len(s_perf)], pipe, parallel_ff, power_constraint))

        if perfs is not None:
            perfs.append(output)
        else:
            perfs = [output]

    def get_required_mem_capacity(self, batch_size, lin, lout):
        """估算给定批大小和序列长度下的总内存需求。
        Args:
            batch_size: 批次中的请求数量。
            lin: 输入序列长度。
            lout: 输出阶段长度。

        Returns:
            以字节为单位的 (weight_memory, kv_memory, temp_memory)。
        """
        ndec = self.model.ndec
        hdim = self.model.hdim
        nhead = self.model.num_heads
        ff_scale = self.model.ff_scale
        w_byte = 2 if self.model.dtype in [DataType.W16A16, DataType.W16A8
                                          ] else 1
        a_byte = 2 if self.model.dtype in [DataType.W16A16, DataType.W8A16
                                          ] else 1
        l = lin + lout - 1

        if 'LLAMA' in self.model.name:
            weight_memory = ndec * hdim * (2 * hdim + 2 * (hdim) +
                                           3 * ff_scale * hdim) * w_byte
        else:
            weight_memory = ndec * hdim * (2 * hdim + 2 * (hdim) +
                                           2 * ff_scale * hdim) * w_byte

        temp_memory = max((hdim + l * nhead) * a_byte, hdim * 2 * a_byte,
                          l * nhead * 2 * a_byte,
                          (ff_scale * hdim + hdim) * a_byte) + l * nhead
        kv_memory = ndec * 2 * l * (hdim) * a_byte #  层数 × K/V两份 × 序列长度 × 隐藏维度 × 每元素字节

        return weight_memory, kv_memory * batch_size, temp_memory * batch_size
