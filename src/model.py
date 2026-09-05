## Define models and layer.
## Generate models
from .type import *
import copy


class Layer:

    def __init__(self, stage, name, type, has_weight, dtype, m, n, k, numOp):
        self.stage = stage
        self.name = name
        self.type = type
        self.has_weight = has_weight
        self.m = m
        self.n = n
        self.k = k
        self.numOp = numOp
        self.dtype = dtype
        self.dbyte = 2
        if dtype in [DataType.W16A16]:
            self.dbyte = 2
        elif dtype in [DataType.W8A8]:
            self.dbyte = 1
        else:
            assert 0, "Only support W16A16, W8A8"
        self.bound = 'compute'  # 'memory'
        self.exec_time = 0
        self.energy = 0

        assert isinstance(type, LayerType), "Not support layer type"
        assert isinstance(dtype, DataType), "Not support data type"

    def  get_infos(self):
        return self.m, self.n, self.k, self.numOp, self.dbyte

    def get_flops(self):
        if self.type == LayerType.SOFTMAX:
            return 5 * self.m * self.n * self.numOp

        elif self.type == LayerType.ACT:
            if 'relu' in self.name:
                return 1 * self.m * self.n * self.numOp
            elif 'glu' in self.name:
                return (8 + 1) * self.m * self.n * self.numOp
            else:
                return 8 * self.m * self.n * self.numOp

        elif self.type == LayerType.NORM:
            return 5 * self.m * self.n * self.numOp

        elif self.type in [LayerType.FC, LayerType.MATMUL]:
            return 2 * self.m * self.n * self.k * self.numOp

        elif self.type in [LayerType.G2G, LayerType.X2G]:
            return 0

        else:
            assert 0, "In Function \"get_flops\": Not support layer type"

    def get_size(self):
        in1 = self.numOp * self.m * self.k * self.dbyte
        in2 = self.numOp * self.n * self.k * self.dbyte
        out = self.numOp * self.m * self.n * self.dbyte

        if self.type in [
                LayerType.SOFTMAX, LayerType.ACT, LayerType.G2G, LayerType.X2G
        ]:
            in1 = self.numOp * self.m * self.n * self.dbyte
            in2 = 0
            out = in1

            # For SwiGLU and GeGLU
            if 'glu' in self.name:
                in2 = in1

        elif self.type == LayerType.NORM:
            in1 = self.numOp * self.m * self.n * self.dbyte
            in2 = in1
            out = in1

        return in1, in2, out


class Transformer:

    def __init__(self, modelinfos, tensor_parallel=8):
        self.sum_decoder = []
        self.gen_decoder = []
        self.name = modelinfos['name']
        self.ndec = modelinfos['ndec'] 
        self.num_heads = modelinfos['num_heads']
        self.hdim = modelinfos['hdim']
        self.ff_scale = modelinfos['ff_scale']
        self.dtype = modelinfos['dtype']
        self.dhead = int(self.hdim / self.num_heads)
        self.tp = tensor_parallel # 目前是gpu_num

    # Prefill 时一次性把历史 prompt 的 K/V 写入异构设备
    # Decode 时每一步只传当前 token 的一小份 K/V：
    def build(self, batch, lin, lout, attn_on_hetero=False):
        '''
        每张 GPU 的输入都是完整的 [B×L,H]。被切分的是 QKV 权重的输出维度：
        完整 X [B,L,H] × 局部 WQKV_i [H,3H/P]=局部 QKV_i [B,L,3H/P], P为GPU个数
        经过 attention 后得到：Context_i = [B,L,H/P], 
        这个 [B,L,H/P] 不是原始 X 的某一块，而是 GPU_i 所负责的那些 attention heads 产生的新结果。
        它和“把 X 沿 H 维切开”恰好形状相同，但数值和语义完全不同。
        随后每张 GPU 用自己的 context 做输出投影：Context_i [B,L,H/P]× WO_i [H/P,H]=Partial_i [B,L,H], WO_i本身是列切
        最后 AllReduce： O = Partial_0 + Partial_1 + ... + Partial_(P-1)
        于是每张 GPU 又得到完整的 O = [B,L,H]

        假设部署模型时已经分片并常驻各 GPU 的本地显存，因此没有看到权重本身切割部署到各个GPU的通信损耗统计
        '''
        self.sum_decoder = [] # prefill
        self.gen_decoder = [] # decode

        # Summarization (for every gpu_i ?)
        self.sum_decoder.append(
            # [B × Lin, H] × [H, 3H/TP], Hidden_dim
            Layer('sum', 'qkv', LayerType.FC, True, self.dtype, batch * lin,
                  3 * int(self.hdim / self.tp), self.hdim, 1))
        if (attn_on_hetero):
            # send kv matrices 异构计算的情况，将计算的kv发送到异构设备一次性存储
            self.sum_decoder.append(
                Layer('sum', 'comm_x2g', LayerType.X2G, False, self.dtype,
                      batch * lin, 2 * int(self.hdim / self.tp), 1, 1))
        self.sum_decoder.append(
            # Q [Lin, dhead] × Kᵀ [dhead, Lin], 执行次数为：batch × 每张 GPU 上的 head 数
            Layer('sum', 'score', LayerType.MATMUL, False, self.dtype, lin,
                  lin, self.dhead,
                  int(self.num_heads / self.tp) * batch))
        self.sum_decoder.append(
            Layer('sum', 'softmax', LayerType.SOFTMAX, False, self.dtype, lin,
                  lin, 1,
                  int(self.num_heads / self.tp) * batch))
        # Attention [Lin, Lin] × V [Lin, dhead]
        self.sum_decoder.append(
            Layer('sum', 'context', LayerType.MATMUL, False, self.dtype, lin,
                  self.dhead, lin,
                  int(self.num_heads / self.tp) * batch))
        # 隐式包含了拼接本 GPU 的 heads？
        # [B × Lin, H/TP] × W [H/TP, H], 相当于左列切右权重行切，
        self.sum_decoder.append(
            Layer('sum', 'proj', LayerType.FC, True, self.dtype, batch * lin,
                  self.hdim, int(self.hdim / self.tp), 1))
        # 根据切割性质进行 AllReduce，得到完整的 O = [B × Lin, H]
        self.sum_decoder.append(
            Layer('sum', 'comm_g2g', LayerType.G2G, False, self.dtype, batch * lin,
                  self.hdim, 1, 1))
        self.sum_decoder.append(
            Layer('sum', 'norm1', LayerType.NORM, False, self.dtype, batch * lin,
                  self.hdim, 1, 1))
        if 'LLAMA' in self.name:
            ''' ff1和ff2采用分GPU的张量并行(列切)
                ff1 ── SiLU ─┐
         X [B × Lin,H]             × ── ff3 ── AllReduce
                ff2 ─────────┘
            '''
            self.sum_decoder.append(
                Layer('sum', 'ff1', LayerType.FC, True, self.dtype, batch * lin,
                      self.ff_scale * int(self.hdim / self.tp), self.hdim, 1))
            self.sum_decoder.append(
                Layer('sum', 'ff2', LayerType.FC, True, self.dtype, batch * lin,
                      self.ff_scale * int(self.hdim / self.tp), self.hdim, 1))
            self.sum_decoder.append(
                Layer('sum', 'glu', LayerType.ACT, False, self.dtype, batch * lin,
                      self.ff_scale * int(self.hdim / self.tp), 1, 1))
            self.sum_decoder.append(
                Layer('sum', 'ff3', LayerType.FC, True, self.dtype, batch * lin,
                      self.hdim, self.ff_scale * int(self.hdim / self.tp), 1))
        else:
            self.sum_decoder.append(
                Layer('sum', 'ff1', LayerType.FC, True, self.dtype, batch * lin,
                      self.ff_scale * int(self.hdim / self.tp), self.hdim, 1))
            if 'OPT' in self.name:
                self.sum_decoder.append(
                    Layer('sum', 'relu', LayerType.ACT, False,
                          self.dtype, batch * lin,
                          self.ff_scale * int(self.hdim / self.tp), 1, 1))
            else:
                self.sum_decoder.append(
                    Layer('sum', 'gelu', LayerType.ACT, False,
                          self.dtype, batch * lin,
                          self.ff_scale * int(self.hdim / self.tp), 1, 1))
            self.sum_decoder.append(
                Layer('sum', 'ff2', LayerType.FC, True, self.dtype, batch * lin,
                      self.hdim, self.ff_scale * int(self.hdim / self.tp), 1))
        self.sum_decoder.append(
            Layer('sum', 'comm_g2g', LayerType.G2G, False, self.dtype, batch * lin,
                  self.hdim, 1, 1))
        self.sum_decoder.append(
            Layer('sum', 'norm2', LayerType.NORM, False, self.dtype, batch * lin,
                  self.hdim, 1, 1))
        # Generation
        for stage in range(1, lout, 1):
            decoder = []
            decoder.append(
                Layer('gen', 'qkv', LayerType.FC, True, self.dtype, batch,
                      3 * int(self.hdim / self.tp), self.hdim, 1))
            if (attn_on_hetero):
                # 其中 Q 用于当前 Attention，K/V 追加进已经存在的 KV cache
                decoder.append(
                    Layer('gen', 'comm_x2g', LayerType.X2G, False, self.dtype,
                          batch, 3 * int(self.hdim / self.tp), 1, 1))
            decoder.append(
                Layer('gen', 'score', LayerType.MATMUL, False, self.dtype, 1,
                      lin + stage, self.dhead,
                      int(self.num_heads / self.tp) * batch))
            decoder.append(
                Layer('gen', 'softmax', LayerType.SOFTMAX, False, self.dtype,
                      1, lin + stage, 1,
                      int(self.num_heads / self.tp) * batch))
            decoder.append(
                Layer('gen', 'context', LayerType.MATMUL, False, self.dtype,
                      1, self.dhead, lin + stage,
                      int(self.num_heads / self.tp) * batch))
            if (attn_on_hetero):
                decoder.append(
                    # Context_i = [B,H/P] 返回GPU
                    Layer('gen', 'comm_x2g', LayerType.X2G, False, self.dtype, 1,
                          self.dhead, 1,
                          int(self.num_heads / self.tp) * batch))
            decoder.append(
                Layer('gen', 'proj', LayerType.FC, True, self.dtype, batch,
                      self.hdim, int(self.hdim / self.tp), 1))
            decoder.append(
                Layer('gen', 'comm_g2g', LayerType.G2G, False, self.dtype, batch,
                      self.hdim, 1, 1))
            decoder.append(
                Layer('gen', 'norm1', LayerType.NORM, False, self.dtype, batch,
                      self.hdim, 1, 1))
            if 'LLAMA' in self.name:
                decoder.append(
                    Layer('gen', 'ff1', LayerType.FC, True, self.dtype, batch,
                          self.ff_scale * int(self.hdim / self.tp), self.hdim,
                          1))
                decoder.append(
                    Layer('gen', 'ff2', LayerType.FC, True, self.dtype, batch,
                          self.ff_scale * int(self.hdim / self.tp), self.hdim,
                          1))
                decoder.append(
                    Layer('gen', 'glu', LayerType.ACT, False, self.dtype, batch,
                          self.ff_scale * int(self.hdim / self.tp), 1, 1))
                decoder.append(
                    Layer('gen', 'ff3', LayerType.FC, True, self.dtype,
                          batch, self.hdim,
                          self.ff_scale * int(self.hdim / self.tp), 1))
            else:
                decoder.append(
                    Layer('gen', 'ff1', LayerType.FC, True, self.dtype, batch,
                          self.ff_scale * int(self.hdim / self.tp), self.hdim,
                          1))
                if 'OPT' in self.name:
                    decoder.append(
                        Layer('gen', 'relu', LayerType.ACT, False,
                              self.dtype, batch,
                              self.ff_scale * int(self.hdim / self.tp), 1, 1))
                else:
                    decoder.append(
                        Layer('gen', 'gelu', LayerType.ACT, False,
                              self.dtype, batch,
                              self.ff_scale * int(self.hdim / self.tp), 1, 1))
                decoder.append(
                    Layer('gen', 'ff2', LayerType.FC, True, self.dtype,
                          batch, self.hdim,
                          self.ff_scale * int(self.hdim / self.tp), 1))

            decoder.append(
                Layer('gen', 'comm_g2g', LayerType.G2G, False, self.dtype, batch,
                      self.hdim, 1, 1))
            decoder.append(
                Layer('gen', 'norm2', LayerType.NORM, False, self.dtype, batch,
                      self.hdim, 1, 1))

            self.gen_decoder.append(copy.deepcopy(decoder))
