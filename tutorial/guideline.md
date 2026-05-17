# triton-ascend 适配 Ascend 的整体梳理

本文基于当前仓库代码做一个工程视角的梳理，重点回答 4 个问题：

1. `triton-ascend` 到底是怎么把 Triton 适配到 Ascend 的。
2. 从 Python kernel 到最终 NPU 可执行二进制，整体编译流程是什么。
3. 仓库里有哪些 Ascend 专用 pass/优化点。
4. 目前进展如何，暴露出的上层问题有哪些。

注意：

- 下面的“进展/问题”主要来自仓库内的 `README`、支持文档、测试、`FIXME`、`skip reason` 和编译脚本，不等同于线上 issue 看板的完整状态。
- 这里说的“Ascend 专用 pass”，既包括真正的 MLIR pass，也包括带 Ascend 语义的后端阶段和编译选项。

## 1. 总体架构：不是重写 Triton，而是在 Triton 上叠三层

从仓库结构看，`triton-ascend` 不是另起炉灶写了一个新编译器，而是在上游 Triton 基础上，叠加了三层适配：

### 1.1 上游 Triton 本体

- 代码基座在 `third_party/triton`
- 负责 Triton 的 Python 前端、JIT、TTIR、通用 pass、运行时框架

### 1.2 Triton patch 层

- 代码在 `triton_patch`
- 通过 `setup.py` 把部分 Python 模块和运行时逻辑覆盖到 `triton/...`
- 这层主要做：
  - 补 Ascend 相关语义约束
  - 修改/扩展前端语言行为
  - 扩展 JIT/runtime/compiler glue code
  - 补部分 IR/类型/算子支持

可以把它理解成：“尽量保持 Triton 使用方式不变，但在前端和 runtime 处补上 Ascend 所需的约束和分支”。

### 1.3 Ascend backend/plugin 层

- 代码主要在 `ascend/backend` 和 `ascend/triton-adapter`
- `setup.py` 通过 `TRITON_PLUGIN_DIRS` 把 `ascend` 注册成 Triton backend
- `ascend/backend/compiler.py` 定义 Ascend 的编译 stage
- `ascend/backend/driver.py` / `npu_utils.cpp` 定义运行时加载和发射方式
- `ascend/triton-adapter` 定义一整套 Triton IR 到 Ascend 可消费 IR 的自定义 pass

这层才是“真正把 Triton IR 改造成 Ascend 编译器能吃的东西”的核心。

## 2. 整体流程：从 Triton kernel 到 Ascend 二进制

可以把整体链路理解成：

`Triton Python Kernel`
-> `Triton 前端/JIT`
-> `TTIR`
-> `Ascend 自定义 adapter pass pipeline`
-> `Linalg/LLVM-MLIR/BishengIR 可消费形态`
-> `bishengir-compile`
-> `NPU binary`
-> `Ascend runtime 加载执行`

下面按仓库中的真实实现拆开。

### 2.1 安装/集成阶段：把 Ascend backend 和 patch 注入 Triton

关键点在 `setup.py`：

- `third_party/triton` 仍然是主包来源
- `triton_patch/python/triton_patch` 会映射到 `triton/triton_patch`
- 同时还会把一些 patch 文件直接映射到上游 Triton 包路径，比如：
  - `triton/compiler/compiler.py`
  - `triton/compiler/code_generator.py`
  - `triton/runtime/jit.py`
  - `triton/runtime/autotuner.py`
  - `triton/testing.py`
- `ascend/backend` 会作为 `triton.backends.ascend` 注入

所以运行时对用户来说仍然像在用 Triton，但实际上前端、runtime、backend 已经被替换/扩展了一层。

### 2.2 运行时识别目标平台

Ascend 运行时入口在：

- `ascend/backend/driver.py`
- `ascend/backend/npu_utils.cpp`

主要做几件事：

- 当前目标识别为 `backend="npu"`
- 通过 `rtGetSocVersion` / 环境变量确定具体架构
- 用 `NPULauncher` 生成 launcher stub
- 用 `npu_utils.cpp` 把编译好的二进制注册到 Ascend runtime

这里有一个 Ascend 特有点：运行时会依赖 `mix_mode` 决定如何注册 devbin。

- `aiv` 时走 `RT_DEV_BINARY_MAGIC_ELF_AIVEC`
- 其他情况走 `RT_DEV_BINARY_MAGIC_ELF`

也就是说，前面编译阶段不仅要生成二进制，还要把 kernel 是 vector/cube/mix 说清楚。

### 2.3 Stage 1：先走一遍 Triton 通用 TTIR 优化

`ascend/backend/compiler.py:make_ttir`

这一步没有太多 Ascend 特化，基本复用 Triton 通用 TTIR pass：

- `add_inliner`
- `ttir.add_combine`
- `canonicalizer`
- `ttir.add_reorder_broadcast`
- `cse`
- `licm`
- `symbol_dce`
- `ttir.add_loop_unroll`

所以 `triton-ascend` 的策略不是从一开始就分叉，而是先吃满一遍 Triton 现成的前端/TTIR 优化。

### 2.4 Stage 2：TTIR 交给 `triton_adapter_opt`

这是适配核心，入口在 `ascend/backend/compiler.py:ttir_to_linalg`。

真实 pass 顺序如下：

1. 可选 `--triton-linearize`
2. `--discrete-mask-access-conversion`
3. `--triton-to-annotation`
4. `--triton-to-unstructure`
5. `--triton-to-hivm`
6. `--triton-to-hfusion`
7. `--triton-to-llvm`
8. `--bubble-up-operation`
9. `--triton-to-linalg`

这说明它的核心思想不是“一步直接从 Triton 到 LLVM”，而是先把 Ascend 难点单独拆平：

- 指针和访存结构问题
- 离散 mask 访问问题
- 非规则访存 scalarize/unstructure 问题
- Ascend 自定义同步语义
- HFusion/HIVM 自定义方言
- 最后再统一收敛到 Linalg

### 2.5 Stage 3：生成 Linalg 之后，交给 Bisheng 编译

NPU 路径下默认不是再走通用 LLVM 后端，而是直接把 `ttadapter.mlir` 交给 `bishengir-compile`：

- `linalg_to_bin_enable_npu_compile_910_95`
- `linalg_to_bin_enable_npu_compile_A2_A3`

这一步会拼接很多 Ascend 编译选项，例如：

- `--enable-auto-multi-buffer`
- `--enable-ubuf-saving`
- `--enable-auto-bind-sub-block`
- `--enable-hivm-auto-cv-balance`
- `--enable-hivm-graph-sync-solver`
- `--enable-hivm-inject-barrier-all-sync`
- `--enable-hivm-inject-block-all-sync`
- `--enable-auto-blockify-loop`
- `--disable-ffts`
- `--enable-debug-info`
- `--enable-sanitizer`

这些选项说明 `triton-ascend` 的一大块能力，其实是“前端生成足够好的 IR + 后端把 Bisheng/NPUCompiler 的优化开关暴露给 Triton 用户/调优器”。

### 2.6 特殊分支：纯 SIMT 模式可直接 TTIR 到二进制

如果 `force_simt_only=True`，则不走 `ttadapter -> linalg`，而是：

`TTIR -> ttir_to_npubin -> bishengir-compile`

并附带：

- `--enable-triton-ir-compile`
- `--pure-simt`
- `--num-warps`
- `--threads-per-warp`

这说明仓库里实际上保留了两条 NPU 路径：

- 主路径：`TTIR -> Adapter -> Linalg/BishengIR -> npubin`
- 特殊路径：`TTIR -> pure-simt compile -> npubin`

## 3. Ascend 专用 pass/优化点

下面按重要性梳理。

## 3.1 `TritonLinearize`

位置：

- `ascend/triton-adapter/lib/TritonLinearize`

作用：

- 先把 Triton pointer-like value 转成“结构化状态”
- 显式引入 offset/stride
- 让 `PtrAnalysis` 可以分析一串 pointer arithmetic
- 把原来隐式的地址计算，线性化成更容易被后续 pass 处理的形式

这是 Ascend 适配里非常关键的一步。因为很多 GPU 风格 Triton kernel 的地址表达式很灵活，但 Ascend 后续编译更需要显式、结构化、可分析的地址状态。

简化理解：

- Triton 原始写法更像“高阶 pointer 运算”
- `TritonLinearize` 负责把它摊平成“base + offsets + strides”的显式表示

## 3.2 `DiscreteMaskAccessConversion`

位置：

- `ascend/triton-adapter/lib/DiscreteMaskAccessConversion`

作用：

- 识别连续 mask 和离散/runtime mask
- 对 `load/store` 做重写
- 避免 mask 场景下的越界读写
- 必要时引入 `sync_block_lock/unlock`

这个 pass 解决的是 Ascend 上非常实际的问题：

- GPU 风格的“先全量 load，再 select/store”在 NPU 上可能引入 OOB 或竞争风险
- 需要把 `contMask` 和 `discMask` 拆开处理

代码里已经明确写了目标：

- 用连续 mask 限定安全访存范围
- 用离散 mask 决定逐元素选择

这是非常典型的 Ascend 适配性 pass。

## 3.3 `TritonToAnnotation`

位置：

- `ascend/triton-adapter/lib/TritonToAnnotation`

作用：

- 把 Triton 的 `AnnotationOp` 转成 Bisheng/Ascend 侧的 `annotation::MarkOp`
- 把前面分析出来的标注信息传给后续 pass

它本身不复杂，但在整条链里承担“分析结果传递器”的作用。

## 3.4 `TritonToUnstructure`

位置：

- `ascend/triton-adapter/lib/TritonToUnstructure`

作用：

- 处理非规则/非结构化访存
- 对部分 tensor 访问做 extract/extract_slice + scalarize
- 把一些不好直接 lower 的访问模式拆成更细粒度操作

这一步本质是在为 Ascend 编译器“降复杂度”。

如果一个 Triton op 的访存模式过于自由，不适合直接映射到 NPU 访存模板，就先拆成更朴素、更可控的形式。

## 3.5 `TritonToHIVM`

位置：

- `ascend/triton-adapter/lib/TritonToHIVM`

作用：

- 把 Triton 里的某些自定义同步操作，映射到 Ascend/HIVM 的 `sync_block_*` 语义
- 显式区分 `cube`/`vector` core 和不同 pipe

这一层是 Ascend 架构语义暴露最直接的地方之一。它说明 `triton-ascend` 不只是换了 backend，而是显式引入了 Ascend 的：

- core 类型
- pipe
- block sync 机制

## 3.6 `TritonToHFusion`

位置：

- `ascend/triton-adapter/lib/TritonToHFusion`

作用：

- 把部分 Triton op 直接转成 HFusion dialect
- 目前代码里能看到的典型例子包括：
  - `mod`
  - `histogram`
  - 带 rounding mode 的 `FpToFp`

这类 pass 的价值在于：

- 某些操作用通用 Linalg 表达不一定最优
- 直接映射到 Ascend 自家方言，给后端留出更多特化空间

## 3.7 `TritonToLLVM`

位置：

- `ascend/triton-adapter/lib/TritonToLLVM`

作用：

- 处理部分 Triton 自定义 op 到 LLVM/中间桥接表示的转换

它不是主角，但说明整条链路里仍然保留了“必要时往 LLVM 语义对齐”的通道。

## 3.8 `BubbleUpOperation`

位置：

- `ascend/triton-adapter/lib/TritonToUnstructure/BubbleUpOperation.cpp`

作用：

- 把 `tensor.extract` / `extract_slice` 往上冒
- 尽量把“先大 tensor 运算，再抽小块”改写成“先抽小块，再做标量/小 tensor 运算”

这个 pass 很工程化，但很有效：

- 能减少后续无谓的大 tensor 转换
- 有助于离散访问场景的局部化处理
- 也方便后续 Linalg/标量化 lowering

## 3.9 `TritonToLinalg`

位置：

- `ascend/triton-adapter/lib/TritonToLinalg`

这是最终的主收敛 pass，作用很多：

- Triton type -> memref/tensor/linalg type 转换
- `load/store/atomic/reduce/scan/sort/gather/dot/...` 等大量 op 转换
- 给函数参数标 `tt.tensor_kind`
- 给函数打上 `mix_mode` 和 `parallel_mode`
- 支持 `named_ops`
- 支持 `enable_select_analysis`
- 支持 `enable_nd2nz_on_vector`

这里有几个 Ascend 特别关键的点：

### `mix_mode`

在 `TritonToLinalgPass.cpp` 里，kernel 会被打成：

- `aiv`
- `mix`

它后面会一路传到 runtime，用来决定 devbin 的注册模式。

### `parallel_mode`

当前能看到的模式至少有：

- `simd`
- `mix_simd_simt`

这说明 `triton-ascend` 不只是“能跑”，而是在编译阶段就显式区分：

- 纯向量核
- 向量 + SIMT 混合核

### `enable_select_analysis`

这个开关和 `SelectCanonicalizer` 连在一起，用于优化 select/mask 相关场景。结合前面的 `DiscreteMaskAccessConversion` 看，mask/条件选择是 Ascend 适配的一条主线。

### `enable_nd2nz_on_vector`

名字已经很说明问题：这是典型 Ascend 数据布局/访存友好化优化点，目标是让后端更容易命中合适的数据格式和访存方式。

## 4. 从工程分层看，triton-ascend 具体做了什么

如果把这套适配抽象一下，本质上做了四类事：

### 4.1 前端约束收紧

通过 `triton_patch/python/triton_patch/language/*`：

- 限定某些 dtype 组合
- 限定某些 API 用法
- 对 Ascend 暂不支持的场景直接报错或给替代路径

典型例子：

- `[fp8, fp64]` 当前不支持
- `tl.dot(..., out_dtype=bfloat16)` 不支持
- `tl.div_rz` / `tl.fmod` / `tl.trunc` 建议改用 `libdevice`

### 4.2 中间表示“结构化”

通过 `linearize / unstructure / bubble-up / annotation / mask conversion`：

- 把 GPU 风格的自由地址计算、自由 mask、自由 tensor 抽取
- 改写成 Ascend 编译器更容易理解和优化的结构化 IR

### 4.3 引入 Ascend 特有语义

通过 `HIVM / HFusion / mix_mode / parallel_mode / tensor_kind`：

- 把 Ascend 的 core、pipe、同步、融合方言、二进制模式显式编码进 IR/metadata

### 4.4 把后端编译器能力暴露出来

通过 `compiler.py` 里的 metadata/option：

- multi-buffer
- ubuf saving
- auto bind sub block
- sync solver
- auto blockify loop
- sanitizer/debug-info

这决定了 `triton-ascend` 不只是一个“兼容层”，也是一个“把 Bisheng/NPU 编译器可调优能力透给 Triton 用户”的封装层。

## 5. 当前进展

按 `README.md` 的路线图，当前仓库给出的公开进展是：

- `2025-05-20`：Triton Ascend 开源
- `2025-06-30`：支持 `85%` Triton Python API，支持连续访存，覆盖基本场景
- `2025-08-15`：补齐 Atomic 类 API，适配 Flaggems 重点算子，提供 Matmul 等高性能样例
- `2025-09-30`：补齐 Scan/Sort 类 API，支持非连续访存，完成 vLLM / SGLang 重点 Triton 算子适配
- `2025-11-14`：`triton-ascend 3.2.0rc4` 预发布

从仓库里的文档和测试覆盖看，可以把“当前进展”再拆成 4 点：

### 5.1 基础平台已经成形

- 已支持平台文档写的是 `Ascend Atlas A2/A3`
- 后端、driver、runtime、adapter、文档、样例、测试都已经齐备
- 从工程完整度看，已经不是 demo，而是一套可持续演进的后端

### 5.2 Python API 支持面已经比较大

`docs/sources/python-api/outline.md` 明确给出：

- Triton op 支持度已经比较高
- 覆盖 creation / shape / memory / reduce / scan / atomic / random / debug 等大类

但很多 op 仍带“数据类型约束”或“形状/语义约束”，说明当前是“广覆盖 + 条件支持”，不是完全无差别兼容。

### 5.3 开源算子适配在推进，但仍分层

`docs/OPLIST.md` 当前明确列出的稳定清单主要还是 Flaggems 一批基础算子：

- `abs/add/div/exp/mul/relu/sigmoid/silu/...`

而文档对 `vllm/sglang` 的表述是“正在逐步支持中”。虽然仓库里已经有大量 `ascend/test/sglang/v0.4.8` 用例，但从对外文档口径看，生态适配仍在推进中。

### 5.4 测试覆盖已经很广

仓库里同时有：

- MLIR conversion lit tests
- `pytest_ut`
- `generalization_cases`
- `benchmark_cases`
- `tutorials`
- `sglang` 适配测试

说明团队当前工作重点之一就是“把支持边界不断外推”。

## 6. 目前暴露出的上层问题

下面这部分不是泛泛而谈，而是从仓库现状里能直接看到的“上层 issue”。

## 6.1 对后端编译器版本强依赖

这是目前最明显的问题之一。

编译脚本里直接写了多处 `FIXME`：

- `ascend/examples/run_test.sh`
- `ascend/examples/run_daily.sh`

能看到的现象包括：

- 某些 CANN 自带 `bishengir-compile` 会“fails lots of cases”
- 需要切换到特定版本工具链
- 还要求配套的 `bisheng compiler`

这说明当前 `triton-ascend` 和底层 NPU 编译器之间仍存在明显的版本耦合，工程上会带来：

- 环境复现成本高
- CI/日测稳定性受工具链版本影响大
- 用户升级 CANN/编译器时风险较高

## 6.2 “支持了”不等于“无限制支持”

虽然 API 覆盖面高，但文档里有大量约束，典型包括：

- `gather` 当前只支持最后一维 `axis=n-1`
- `permute/trans` 不支持不相邻轴转置
- `atomic_*` 在 loop 中有限制
- `tensor_descriptor` 只支持绑定式使用
- `mod` 的 `int64` 范围有限
- `ALL tensor` 总片上空间仍受 `96KB/192KB` 约束

这说明当前主要问题已经不是“完全不能用”，而是“很多场景能用，但需要按 Ascend 规则改写 kernel”。

## 6.3 大规模 kernel 的核心矛盾仍然是 `coreDim` 和 UB

迁移指南已经把这件事写得很明确：

- `coreDim <= 65535`
- UB 容量有限

这意味着很多 GPU 风格 kernel 直接迁移到 NPU 时，最常见的不是语法错，而是：

- grid 太大
- tile 太大
- 局部 buffer 太大

所以 `triton-ascend` 当前仍然强依赖人工或 autotune 做：

- block size 调整
- 子块切分
- 流水化/multi-buffer 配置

本质上，这是 Ascend 架构约束带来的“性能适配问题”，也是上层迁移成本的主要来源。

## 6.4 mask / 非规则访存仍是薄弱点

从 `DiscreteMaskAccessConversion` 的存在方式，以及测试中的 skip reason 看，这块还在持续收敛。

典型信号：

- `test_linearize.py` 里有：
  - `mask load still has issues to be fixed by bisheng`
  - `mask still has issues to be fixed`

这说明虽然仓库已经专门做了 discrete mask 相关 pass，但：

- 非规则 mask
- linearize + mask 叠加场景
- 部分行为仍依赖底层 compiler 进一步修复

这是目前最典型的“IR 适配层已经有方案，但后端编译器还没完全兜住”的问题。

## 6.5 调试/打印/断言类能力还不完全成熟

从测试 skip reason 能看到：

- `waiting for TA to support`
- `waiting for compiler to support`
- `waiting for bishengir-compile to support`

集中出现在：

- `device_print`
- `static_print/static_assert` 相关测试
- 部分 `pow`、`gather` 等场景

这说明：

- 核心算子编译链优先级更高
- 调试可观测性、脚本化打印、断言这类“开发者体验能力”仍在补齐

## 6.6 某些算子/测试仍有稳定性和精度波动

从测试里能直接看到一些 case 被标成：

- `randomly failed`
- `randomly failed accuracy test`

例如：

- `tan`
- `softmax`
- `max/min vector` 相关 case

这类问题通常不是“功能完全不支持”，而是：

- 后端生成不够稳定
- 数值路径在某些输入/切分下不稳定
- 工具链版本切换后行为会漂移

这对于上层框架接入来说，意味着“要看平均可用性，也要看长尾稳定性”。

## 6.7 生态适配还处在“重点算子突破”阶段

虽然仓库里已经有不少 `sglang` 测试，README 也提到 vLLM / SGLang，但从 `OPLIST.md` 的公开口径看：

- 目前真正对外明确列出的仍主要是 Flaggems 基础算子
- 更复杂的大模型推理 kernel 仍在逐步补齐

所以当前生态适配状态更准确的说法是：

- “核心路径已打通”
- “重点算子已在推进”
- “全面生态兼容还没完全收口”

## 7. 一个简化结论

如果只用一句话概括：

`triton-ascend` 的做法，不是把 Triton 直接翻译成 Ascend 指令，而是先在 Triton 和 Bisheng/NPUCompiler 之间插入一层很厚的 adapter，把 GPU 风格的 Triton IR 改写成更结构化、更显式、更符合 Ascend core/pipe/memory/sync 语义的 IR，再交给后端编译器完成最终生成。

如果再拆成三句：

1. 上层尽量保持 Triton 用法不变。
2. 中间通过 `linearize + mask conversion + unstructure + HIVM/HFusion + Linalg` 把 IR 改造成 Ascend 友好形式。
3. 底层大量依赖 `bishengir-compile` 和运行时元数据，把 `mix_mode / parallel_mode / multi-buffer / sync` 等 Ascend 特性真正落地。

## 8. 关键代码入口索引

后续如果你想继续深挖，建议按下面顺序读：

- `setup.py`
- `ascend/backend/compiler.py`
- `ascend/backend/driver.py`
- `ascend/backend/npu_utils.cpp`
- `ascend/triton-adapter/lib/TritonLinearize`
- `ascend/triton-adapter/lib/DiscreteMaskAccessConversion`
- `ascend/triton-adapter/lib/TritonToUnstructure`
- `ascend/triton-adapter/lib/TritonToHIVM`
- `ascend/triton-adapter/lib/TritonToHFusion`
- `ascend/triton-adapter/lib/TritonToLinalg`
- `docs/sources/python-api/outline.md`
- `docs/sources/programming-guide/migration.md`
- `docs/OPLIST.md`

