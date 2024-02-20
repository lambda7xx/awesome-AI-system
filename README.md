# Awesome AI System

This repo is motivated by [awesome tensor compilers](https://github.com/merrymercy/awesome-tensor-compilers.git).
## Contents

- [Paper-Code](#paper-code)
  - [LLM Serving Framework](#llm-serving-framework)
  - [LLM Inference(System Side)](#LLM-System-Side)
  - [LLM Inference(AI Side)](#LLM-AI-Side)
  - [LLM Platform](#LLM-Platform)
  - [LoRA](#LoRA)
  - [Framework](#framework)
  - [Parallellism Training](#parallellism-training)
  - [Training](#training)
  - [Communication](#communication)
  - [Serving-Inference](#Serving-Inference)
  - [MoE](#MoE)
  - [GPU Cluster Management](#gpu-cluster-management)
  - [Schedule and Resource Management](#schedule)
  - [Optimization](#optimzation)
  - [GNN](#GNN)
  - [Fine-Tune](#Fine-Tune)
  - [Energy](#energy)
  - [Misc](#Misc)
- [Contribute](#contribute)

## Paper-Code

### LLM Serving Framework

| Title | Github|
|:-----:|:-----:|
| MLC LLM| [![Star](https://img.shields.io/github/stars/mlc-ai/mlc-llm.svg)](https://github.com/mlc-ai/mlc-llm/) |
| TensorRT-LLM | [![Star](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg)](https://github.com/NVIDIA/TensorRT-LLM.git) |
| xFasterTransformer |  [![Star](https://img.shields.io/github/stars/intel/xFasterTransformer.svg)](https://github.com/intel/xFasterTransformer)|
| CTranslate2(low latency) | [![Star](https://img.shields.io/github/stars/OpenNMT/CTranslate2.svg)](https://github.com/OpenNMT/CTranslate2.git)|


<!-- [![Star](xFasterTransformer(CPU Side)](https://github.com/intel/xFasterTransformer)| -->
<!-- 
- [TensorRT-LLM by nvidia ](https://github.com/NVIDIA/TensorRT-LLM.git)
- [CTranslate2(low latency)](https://github.com/OpenNMT/CTranslate2.git) -->

### LLM Inference (System Side)

| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| Efficient Memory Management for Large Language Model Serving with PagedAttention| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2309.06180.pdf) | [![Star](https://img.shields.io/github/stars/vllm-project/vllm.svg)](https://github.com/vllm-project/vllm.git) | - | SOSP'23 |
| SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.09781.pdf) | [![Star](https://img.shields.io/github/stars/flexflow/FlexFlow.svg)](https://github.com/flexflow/FlexFlow) | - | Dec,2023 |
|Liger: Interleaving Intra- and Inter-Operator Parallelism for Distributed Large Model Inference| - | [![Star](https://img.shields.io/github/stars/MachineLearningSystem/24PPOPP-Liger.svg)](https://github.com/MachineLearningSystem/24PPOPP-Liger) |-| PPOPP'24
|Efficiently Programming Large Language Models using SGLang| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2312.07104.pdf)| [![Star](https://img.shields.io/github/stars/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang.git) | - | Dec, 2023 | 
| Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.vldb.org/pvldb/vol17/p211-xia.pdf) | [![Star](https://img.shields.io/github/stars/AlibabaResearch/flash-llm.svg)](https://github.com/AlibabaResearch/flash-llm) | - | VLDB'24 |
| PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf) | [![Star](https://img.shields.io/github/stars/SJTU-IPADS/PowerInfer.svg)](https://github.com/SJTU-IPADS/PowerInfer) | - | Dec, 2023 |

### LLM Inference(AI Side)
| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| BitDelta: Your Fine-Tune May Only Be Worth One Bit| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.10193) | [![Star](https://img.shields.io/github/stars/FasterDecoding/BitDelta.svg)](https://github.com/FasterDecoding/BitDelta.git) | - | Feb,2024 |
| Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.10774) | [![Star](https://img.shields.io/github/stars/FasterDecoding/Medusa.svg)](https://github.com/FasterDecoding/Medusa.git) | - | Jan,2024 |
| LLMCompiler: An LLM Compiler for Parallel Function Calling| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2312.04511.pdf) | [![Star](https://img.shields.io/github/stars/SqueezeAILab/LLMCompiler.svg)](https://github.com/SqueezeAILab/LLMCompiler.git) | - | Dec,2023 |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2312.00752.pdf) | [![Star](https://img.shields.io/github/stars/state-spaces/mamba.svg)](https://github.com/state-spaces/mamba.git) | - | Dec,2023 |
| Teaching LLMs memory management for unbounded context| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.08560) | [![Star](https://img.shields.io/github/stars/cpacker/MemGPT.svg)](https://github.com/cpacker/MemGPT.git) | - | Oct,2023 |
| Break the Sequential Dependency of LLM Inference Using Lookahead Decoding| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.02057) | [![Star](https://img.shields.io/github/stars/hao-ai-lab/LookaheadDecoding.svg)](https://github.com/hao-ai-lab/LookaheadDecoding.git) | - | Feb,2024 |
| EAGLE: Lossless Acceleration of LLM Decoding by Feature Extrapolation| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2401.15077.pdf) | [![Star](https://img.shields.io/github/stars/SafeAILab/EAGLE.svg)](https://github.com/SafeAILab/EAGLE.git) | - | Jan,2024 |

### LLM Platform

| Title | Github| Website
|:-----:|:-----:|:-----:|
| FastChat | [![Star](https://img.shields.io/github/stars/lm-sys/FastChat.svg)](https://github.com/lm-sys/FastChat.git)| [![Website](https://img.shields.io/badge/Website-9cf)](https://chat.lmsys.org/) |


### LoRA

| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| S-LoRA: Serving Thousands of Concurrent LoRA Adapters| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2311.03285.pdf) | [![Star](https://img.shields.io/github/stars/S-LoRA/S-LoRA.svg)](https://github.com/S-LoRA/S-LoRA.git) | - | Nov,2023 |
| Punica: Serving multiple LoRA finetuned LLM as one| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2310.18547.pdf) | [![Star](https://img.shields.io/github/stars/punica-ai/punica.svg)](https://github.com/punica-ai/punica.git) | - | Oct,2023 |





### Framework
- code [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning OSDI'22](https://github.com/alpa-projects/alpa.git) 

  paper [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning OSDI'22](https://arxiv.org/pdf/2201.12023.pdf)

- code [Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization OSDI'22 ](https://github.com/flexflow/FlexFlow) OSDI'22 
   
  paper [Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization OSDI'22](https://www.usenix.org/system/files/osdi22-unger.pdf)

- code [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM SC21 ](https://github.com/NVIDIA/Megatron-LM.git) 

  paper [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM SC21 ](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf)

- code [A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters OSDI'20](https://github.com/bytedance/byteps) 
  
  paper [A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters OSDI'20](https://www.usenix.org/system/files/osdi20-jiang.pdf)

- code [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training ICPP'23](https://github.com/hpcaitech/ColossalAI)

  paper [Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training ICPP'23](https://dl.acm.org/doi/pdf/10.1145/3605573.3605613)

- code [HET: Scaling out Huge Embedding Model Training via Cache-enabled Distributed Framework VLDB'22](https://github.com/MachineLearningSystem/Hetu) 
 
  paper [HET: Scaling out Huge Embedding Model Training via Cache-enabled Distributed Framework VLDB'22](https://www.vldb.org/pvldb/vol15/p312-miao.pdf)

### Parallellism Training
- code [zero-bubble-pipeline-parallelism](https://github.com/sail-sg/zero-bubble-pipeline-parallelism)
  
  paper [NEAR ZERO BUBBLE PIPELINE PARALLELISM ICLR'24](https://openreview.net/pdf?id=tuzTN0eIO5 )
- code [ DynaPipe: Optimizing Multi-task Training through Dynamic Pipelines Eurosys'24](https://github.com/MachineLearningSystem/24Eurosys-optimizing-multitask-training-through-dynamic-pipelines)
  
  paper [ DynaPipe: Optimizing Multi-task Training through Dynamic Pipelines Eurosys'24](https://assets.amazon.science/33/e5/023653cb46d9abb4baa576c571b3/dynapipe-optimizing-multi-task-training-through-dynamic-pipelines.pdf)
- [Aceso: Efficient Parallel DNN Training through Iterative Bottleneck Alleviation Eurosys'24](https://github.com/MachineLearningSystem/24Eurosys-Aceso)
- code [HAP: SPMD DNN Training on Heterogeneous GPU Clusters with Automated Program Synthesis Eurosys'24](https://github.com/MachineLearningSystem/24Eurosys-hap)
  
  paper [HAP: SPMD DNN Training on Heterogeneous GPU Clusters with Automated Program Synthesis Eurosys'24](https://i.cs.hku.hk/~cwu/papers/swzhang-eurosys24.pdf)

- code [Calculon: A Methodology and Tool for High-Level Co-Design of Systems and Large Language Models SC'23](https://github.com/MachineLearningSystem/23sc-calculon)

  paper [Calculon: A Methodology and Tool for High-Level Co-Design of Systems and Large Language Models SC'23](https://dl.acm.org/doi/pdf/10.1145/3581784.3607102)

- code [PipeFisher: Efficient Training of Large Language Models Using Pipelining and Fisher Information Matrices  MLSYS'23](https://github.com/MachineLearningSystem/23MLSYS-pipe-fisher)

 
  paper [PipeFisher: Efficient Training of Large Language Models Using Pipelining and Fisher Information Matrices  MLSYS'23](https://arxiv.org/pdf/2211.14133.pdf)

- code [Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs NSDI'23 ](https://github.com/MachineLearningSystem/bamboo)  
 
  paper [Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs NSDI'23 ](https://www.usenix.org/system/files/nsdi23-thorpe.pdf)

- code [MPress: Democratizing Billion-Scale Model Training on Multi-GPU Servers via Memory-Saving Inter-Operator Parallelism HPCA'23 ](https://github.com/MachineLearningSystem/HPCA23-mpress) 
 
  paper [MPress: Democratizing Billion-Scale Model Training on Multi-GPU Servers via Memory-Saving Inter-Operator Parallelism HPCA'23 ](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10071077)

- code [Optimus-CC: Efficient Large NLP Model Training with 3D Parallelism Aware Communication Compression ASPLOS'23](https://github.com/MachineLearningSystem/Optimus-CC) 
 
  paper [Optimus-CC: Efficient Large NLP Model Training with 3D Parallelism Aware Communication Compression ASPLOS'23](https://arxiv.org/pdf/2301.09830.pdf)
- code [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning OSDI'22](https://github.com/alpa-projects/alpa.git) 

  paper [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning OSDI'22](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf)

- code [AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness NeurIPS '22 ](https://github.com/MachineLearningSystem/AMP) 

  paper [AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness NeurIPS '22 ](https://arxiv.org/pdf/2210.07297.pdf)

- code [Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization OSDI'22](https://github.com/flexflow/FlexFlow) 

  paper [Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization OSDI'22](https://www.usenix.org/system/files/osdi22-unger.pdf)

- code [NASPipe: High Performance and Reproducible Pipeline Parallel Supernet Training via Causal Synchronous Parallelism](https://github.com/MachineLearningSystem/naspipe) ASPLOS'22

  paper [NASPipe: High Performance and Reproducible Pipeline Parallel Supernet Training via Causal Synchronous Parallelism](https://dl.acm.org/doi/pdf/10.1145/3503222.3507735)
- code [Varuna: Scalable, Low-cost Training of Massive Deep Learning Models](https://github.com/MachineLearningSystem/varuna) Eurosys'22 

  paper [Varuna: Scalable, Low-cost Training of Massive Deep Learning Models](https://dl.acm.org/doi/pdf/10.1145/3492321.3519584)

- code [Chimera: efficiently training large-scale neural networks with bidirectional pipelines SC'21 ](https://github.com/MachineLearningSystem/Chimera) 
 
  paper [Chimera: efficiently training large-scale neural networks with bidirectional pipelines SC'21 ](https://dl.acm.org/doi/pdf/10.1145/3458817.3476145)

- code [Piper: Multidimensional Planner for DNN Parallelization NeurIPS'21](https://github.com/MachineLearningSystem/piper) 

  paper [Piper: Multidimensional Planner for DNN Parallelization NeurIPS'21](https://proceedings.neurips.cc/paper_files/paper/2021/file/d01eeca8b24321cd2fe89dd85b9beb51-Paper.pdf)

- code [PipeTransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models  ICML'21](https://github.com/MachineLearningSystem/PipeTransformer.git)

  paper [PipeTransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models  ICML'21](http://proceedings.mlr.press/v139/he21a/he21a.pdf)

- code [DAPPLE: An Efficient Pipelined Data Parallel Approach for Large Models Training PPOPP'21](https://github.com/MachineLearningSystem/DAPPLE)

  paper [DAPPLE: An Efficient Pipelined Data Parallel Approach for Large Models Training PPOPP'21](https://dl.acm.org/doi/pdf/10.1145/3437801.3441593)

- code [TeraPipe:Large-Scale Language Modeling with Pipeline Parallelism ICML'21 ](https://github.com/MachineLearningSystem/terapipe) 

  paper [TeraPipe:Large-Scale Language Modeling with Pipeline Parallelism ICML'21 ](https://danyangzhuo.com/papers/ICML21-TeraPipe.pdf)

- code [PipeDream: Pipeline Parallelism for DNN Training SOSP'19 ](https://github.com/MachineLearningSystem/pipedream.git) 

  paper [PipeDream: Pipeline Parallelism for DNN Training SOSP'19 ](https://people.eecs.berkeley.edu/~matei/papers/2019/sosp_pipedream.pdf)

- code [SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient](https://github.com/MachineLearningSystem/swarm)
 

  paper [SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient](https://proceedings.mlr.press/v202/ryabinin23a/ryabinin23a.pdf)

- code [Merak: An Efficient Distributed DNN Training Framework with Automated 3D Parallelism for Giant Foundation Models](https://github.com/MachineLearningSystem/Merak)

  paper [Merak: An Efficient Distributed DNN Training Framework with Automated 3D Parallelism for Giant Foundation Models](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10049507)

- [awesome distributed deep learning](https://github.com/MachineLearningSystem/Awesome-Distributed-Deep-Learning.git)
- [awsome parallelism](https://github.com/MachineLearningSystem/awesome-Auto-Parallelism)

### Training 

- code [ModelKeeper: Accelerating DNN Training via Automated Training Warmup NSDI'23](https://github.com/MachineLearningSystem/ModelKeeper) 

  paper [ModelKeeper: Accelerating DNN Training via Automated Training Warmup NSDI'23](https://www.usenix.org/system/files/nsdi23-lai-fan.pdf)


- code [STRONGHOLD: Fast and Affordable Billion-scale Deep Learning Model Training SC'22](https://github.com/MachineLearningSystem/sc22-ae-big_model) 

  paper  [STRONGHOLD: Fast and Affordable Billion-scale Deep Learning Model Training SC'22](https://dl.acm.org/doi/pdf/10.5555/3571885.3571979)

- code [Whale: Efficient Giant Model Training over Heterogeneous {GPUs}ATC'22 ](https://github.com/MachineLearningSystem/EasyParallelLibrary) 

  paper [Whale: Efficient Giant Model Training over Heterogeneous {GPUs}ATC'22 ](https://www.usenix.org/system/files/atc22-jia-xianyan.pdf)

- code [GeePS: Scalable Deep Learning on Distributed GPUs with a GPU-Specialized Parameter Server Eurosys'16](https://github.com/MachineLearningSystem/geeps) 

  paper [GeePS: Scalable Deep Learning on Distributed GPUs with a GPU-Specialized Parameter Server Eurosys'16](https://www.pdl.cmu.edu/PDL-FTP/CloudComputing/GeePS-cui-eurosys16.pdf)


### Communication
- code [ARK: GPU-driven Code Execution for Distributed Deep Learning NSDI'23](https://github.com/MachineLearningSystem/23NSDI-arkwo)

  paper [ARK: GPU-driven Code Execution for Distributed Deep Learning NSDI'23](https://www.usenix.org/system/files/nsdi23-hwang.pdf)

- code [TopoOpt: Optimizing the Network Topology for Distributed DNN Training NSDI'23 ](https://github.com/MachineLearningSystem/TopoOpt) 

   paper [TopoOpt: Optimizing the Network Topology for Distributed DNN Training NSDI'23 ](https://www.usenix.org/system/files/nsdi23-wang-weiyang.pdf)

- code [Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads ASPLOS'22 ](https://github.com/parasailteam/coconet.git) 

  paper [Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads ASPLOS'22 ](https://dl.acm.org/doi/pdf/10.1145/3503222.3507778)

- code [Efficient Sparse Collective Communication and its application to Accelerate Distributed Deep Learning SIGCOMM'21 ](https://github.com/MachineLearningSystem/omnireduce.git) 

  paper [Efficient Sparse Collective Communication and its application to Accelerate Distributed Deep Learning SIGCOMM'21 ](https://dl.acm.org/doi/pdf/10.1145/3452296.3472904)

### Serving-Inference

- code [Paella: Low-latency Model Serving with Virtualized GPU Scheduling SOSP'23](https://github.com/MachineLearningSystem/23sosp-paella)

  paper [Paella: Low-latency Model Serving with Virtualized GPU Scheduling SOSP'23](https://dl.acm.org/doi/pdf/10.1145/3600006.3613163)

- code [AlpaServe: Statistical Multiplexing with Model  Parallelism for Deep Learning Serving OSDI'23](https://github.com/MachineLearningSystem/OSDI23-mms)
  
  paper [AlpaServe: Statistical Multiplexing with Model  Parallelism for Deep Learning Serving OSDI'23](https://www.usenix.org/system/files/osdi23-li-zhuohan.pdf)
- code [Optimizing Dynamic Neural Networks with Brainstorm OSDI'23](https://github.com/MachineLearningSystem/23OSDI-brainstorm)

  paper [Optimizing Dynamic Neural Networks with Brainstorm OSDI'23](https://www.usenix.org/system/files/osdi23-cui.pdf)

- code [Fast and Efficient Model Serving Using Multi-GPUs with Direct-Host-Access Eurosys'23](https://github.com/MachineLearningSystem/DeepPlan.git) 

  paper [Fast and Efficient Model Serving Using Multi-GPUs with Direct-Host-Access Eurosys'23](https://jeongseob.github.io/papers/jeong_eurosys23.pdf)


- code [Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs.ASPLOS'23](https://github.com/MachineLearningSystem/hidet)

  paper [Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs.ASPLOS'23](https://arxiv.org/pdf/2210.09603.pdf)
- code [MPCFormer: fast, performant, and private transformer inference with MPC ICLR'23](https://github.com/DachengLi1/MPCFormer)  

  paper [MPCFormer: fast, performant, and private transformer inference with MPC ICLR'23](https://arxiv.org/pdf/2211.01452.pdf)

- code [High-throughput Generative Inference of Large Language Modelwith a Single GPU ICML'23](https://github.com/MachineLearningSystem/FlexGen) 
 
  paper [High-throughput Generative Inference of Large Language Modelwith a Single GPU ICML'23](https://arxiv.org/pdf/2303.06865.pdf)

- code [VELTAIR: Towards High-Performance Multi-Tenant Deep Learning Serving via Adaptive Compilation and Scheduling ASPLOS'22](https://github.com/MachineLearningSystem/VELTAIR_ASPLOS22) 

  paper [VELTAIR: Towards High-Performance Multi-Tenant Deep Learning Serving via Adaptive Compilation and Scheduling ASPLOS'22](https://arxiv.org/pdf/2201.06212.pdf)

- code [DVABatch: Diversity-aware Multi-Entry Multi-Exit Batching for Efficient Processing of DNN Services on GPUs ATC'22 ](https://github.com/MachineLearningSystem/DVABatch)  


  paper [DVABatch: Diversity-aware Multi-Entry Multi-Exit Batching for Efficient Processing of DNN Services on GPUs ATC'22 ](https://www.usenix.org/system/files/atc22-cui.pdf)

- code [Cocktail: A Multidimensional Optimization for Model Serving in Cloud NSDI'22 ](https://github.com/MachineLearningSystem/cocktail) 

  paper [Cocktail: A Multidimensional Optimization for Model Serving in Cloud NSDI'22 ](https://www.usenix.org/system/files/nsdi22-paper-gunasekaran.pdf)

- code [Serving Heterogeneous Machine Learning Models on Multi-GPU Servers with Spatio-Temporal Sharing ATC'22](https://github.com/MachineLearningSystem/glet) 

  paper [Serving Heterogeneous Machine Learning Models on Multi-GPU Servers with Spatio-Temporal Sharing ATC'22](https://www.usenix.org/system/files/atc22-choi-seungbeom.pdf)

- code [RIBBON: cost-effective and qos-aware deep learning model inference using a diverse pool of cloud computing instances SC'21](https://github.com/MachineLearningSystem/SC21_Ribbon) 

  paper [RIBBON: cost-effective and qos-aware deep learning model inference using a diverse pool of cloud computing instances SC'21](https://dl.acm.org/doi/pdf/10.1145/3458817.3476168)

- code [INFaaS: Automated Model-less Inference Serving ATC'21 ](https://github.com/MachineLearningSystem/INFaaS.git)

  paper  [INFaaS: Automated Model-less Inference Serving ATC'21 ](https://www.usenix.org/system/files/atc21-romero.pdf)

- code [Enable Simultaneous DNN Services Based on Deterministic Operator Overlap and Precise Latency Prediction SC'21](https://github.com/MachineLearningSystem/Abacus) 

  paper [Enable Simultaneous DNN Services Based on Deterministic Operator Overlap and Precise Latency Prediction SC'21](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9910118)


- code [Serving DNNs like Clockwork: Performance Predictability from the Bottom Up OSDI'20](https://github.com/MachineLearningSystem/clockwork) 

  paper [Serving DNNs like Clockwork: Performance Predictability from the Bottom Up OSDI'20](https://www.usenix.org/system/files/osdi20-gujarati.pdf)

- code [Exploiting Cloud Services for Cost-Effective, SLO-Aware Machine Learning Inference Serving ATC'19 ](https://github.com/MachineLearningSystem/MArk-Project) 

  paper [Exploiting Cloud Services for Cost-Effective, SLO-Aware Machine Learning Inference Serving ATC'19 ](https://www.usenix.org/system/files/atc19-zhang-chengliang.pdf)

- code [Nexus: a GPU cluster engine for accelerating DNN-based video analysis SOSP'19 ](https://github.com/MachineLearningSystem/nexus) 

  paper [Nexus: a GPU cluster engine for accelerating DNN-based video analysis SOSP'19 ](https://homes.cs.washington.edu/~arvind/papers/nexus.pdf)

- code [Clipper:A low-latency prediction-serving system NSDI'17](https://github.com/ucbrise/clipper) 

  paper [Clipper:A low-latency prediction-serving system NSDI'17](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf)


### MoE
- code [SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Static and Dynamic Parallelization ATC'23](https://github.com/MachineLearningSystem/23ATC-SmartMoE-AE)

  paper [SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Static and Dynamic Parallelization ATC'23](https://www.usenix.org/system/files/atc23-zhai.pdf)
- code [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts MLSYS'23 ](https://github.com/stanford-futuredata/megablocks) 

  paper [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts MLSYS'23 ](https://arxiv.org/pdf/2211.15841.pdf)
  
- code [Tutel: Adaptive Mixture-of-Experts at Scale MLSYS'23](https://github.com/MachineLearningSystem/tutel-MOE) 

  paper [Tutel: Adaptive Mixture-of-Experts at Scale MLSYS'23](https://arxiv.org/pdf/2206.03382.pdf)

- code [FastMoE: A Fast Mixture-of-Expert Training System PPOPP'22](https://github.com/MachineLearningSystem/fastmoe-thu) 

  paper [FastMoE: A Fast Mixture-of-Expert Training System PPOPP'22](https://dl.acm.org/doi/pdf/10.1145/3503221.3508418)

- code [AutoMoE: Neural Architecture Search for Efficient Sparsely Activated Transformers ICLR'23](https://github.com/MachineLearningSystem/AutoMoE) 


  paper [AutoMoE: Neural Architecture Search for Efficient Sparsely Activated Transformers ICLR'23](https://openreview.net/pdf?id=3yEIFSMwKBC)

- [awesome MoE](https://github.com/MachineLearningSystem/awesome-mixture-of-experts)

- [MoE Paper](https://github.com/MachineLearningSystem/Awesome-Mixture-of-Experts-Papers)



### GPU Cluster Management
- code [Lucid: A Non-Intrusive, Scalable and Interpretable Scheduler for Deep Learning Training Jobs ASPLOS'23](https://github.com/MachineLearningSystem/Lucid) 

  paper [Lucid: A Non-Intrusive, Scalable and Interpretable Scheduler for Deep Learning Training Jobs ASPLOS'23](https://dl.acm.org/doi/pdf/10.1145/3575693.3575705)

- code [Shockwave: Fair and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning NSDI'23](https://github.com/MachineLearningSystem/shockwave)  

  paper [Shockwave: Fair and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning NSDI'23](https://www.usenix.org/system/files/nsdi23-zheng.pdf)

- code [Synergy : Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters OSDI'22](https://github.com/MachineLearningSystem/synergy.git) 

  paper [Synergy : Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters OSDI'22](https://www.usenix.org/system/files/osdi22-mohan.pdf)

- code [Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning OSDI'21](https://github.com/MachineLearningSystem/adaptdl) 

  paper [Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning OSDI'21](https://www.usenix.org/system/files/osdi21-qiao.pdf)

- code [Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads OSDI'20](https://github.com/MachineLearningSystem/gavel)

  paper [Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads OSDI'20](https://www.usenix.org/system/files/osdi20-narayanan_deepak.pdf)

- code [Tiresias -- A GPU Cluster Manager for Distributed Deep Learning Training without complete job information  NSDI'19](https://github.com/MachineLearningSystem/Tiresias)


  paper [Tiresias -- A GPU Cluster Manager for Distributed Deep Learning Training without complete job information  NSDI'19](https://www.usenix.org/system/files/nsdi19-gu.pdf)

- code [Chronus: A Novel Deadline-aware Scheduler for Deep Learning Training Jobs SOCC'21 ](https://github.com/MachineLearningSystem/ChronusArtifact) 

  paper [Chronus: A Novel Deadline-aware Scheduler for Deep Learning Training Jobs SOCC'21 ](https://yezhisheng.me/publication/chronus/chronus_preprint.pdf)

- [awesome DL scheduler](https://github.com/MachineLearningSystem/Awesome-DL-Scheduling-Papers.git)

### Schedule and Resource Management
- code [An interference-aware scheduler for fine-grained GPU sharing Resources Eurosys'24](https://github.com/MachineLearningSystem/24Eurosys-orion.git)

  paper [An interference-aware scheduler for fine-grained GPU sharing Resources Eurosys'24](https://anakli.inf.ethz.ch/papers/orion_eurosys24.pdf)


- code [ElasticFlow: An Elastic Serverless Training Platform for Distributed Deep Learning ASPLOS'23](https://github.com/MachineLearningSystem/ElasticFlow-ASPLOS23) 

  paper [ElasticFlow: An Elastic Serverless Training Platform for Distributed Deep Learning ASPLOS'23](https://cp5555.github.io/publications/elasticflow-asplos23.pdf)

- code [Multi-Resource Interleaving for Deep Learning Training SIGCOMM'22](https://github.com/MachineLearningSystem/Muri) 
  
  paper [Multi-Resource Interleaving for Deep Learning Training SIGCOMM'22](https://xinjin.github.io/files/SIGCOMM22_Muri.pdf)

- code [Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training ASPLOS'24](https://github.com/MachineLearningSystem/slapo)  

  paper [Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training ASPLOS'24](https://arxiv.org/pdf/2302.08005.pdf)

- code [Out-of-order backprop: an effective scheduling technique for deep learning Eurosys'22 ](https://github.com/MachineLearningSystem/ooo-backprop) 

  paper [Out-of-order backprop: an effective scheduling technique for deep learning Eurosys'22 ](https://dl.acm.org/doi/pdf/10.1145/3492321.3519563)

- code [KungFu: Making Training in Distributed Machine Learning Adaptive OSDI'20](https://github.com/MachineLearningSystem/KungFu) 
 
  paper [KungFu: Making Training in Distributed Machine Learning Adaptive OSDI'20](https://www.usenix.org/system/files/osdi20-mai.pdf)

- code [PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications OSDI'20 ](https://github.com/MachineLearningSystem/PipeSwitch) 

  paper [PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications OSDI'20 ](https://www.usenix.org/system/files/osdi20-bai.pdf)


### Optimization
- code [GLake: optimizing GPU memory management and IO transmission ASPLOS'24](https://github.com/MachineLearningSystem/24ASPLOS-glake)
- code [Spada: Accelerating Sparse Matrix Multiplication with Adaptive Dataflow ASPLOS'23 ](https://github.com/MachineLearningSystem/spada-sim) 

  paper [Spada: Accelerating Sparse Matrix Multiplication with Adaptive Dataflow ASPLOS'23 ](https://dl.acm.org/doi/pdf/10.1145/3575693.3575706)

- code [MISO: Exploiting Multi-Instance GPU Capability on Multi-Tenant GPU Clusters SOCC'22 ](https://github.com/MachineLearningSystem/socc22-miso) 
  
  paper [MISO: Exploiting Multi-Instance GPU Capability on Multi-Tenant GPU Clusters SOCC'22 ](https://dspace.mit.edu/bitstream/handle/1721.1/147687/3542929.3563510.pdf?sequence=1&isAllowed=y)

- code [Accpar: Tensor partitioning for heterogeneous deep learning accelerators HPCA'20 ](https://github.com/MachineLearningSystem/AccPar) 

  paper [Accpar: Tensor partitioning for heterogeneous deep learning accelerators HPCA'20 ](http://alchem.usc.edu/portal/static/download/accpar.pdf)


- code [Hidet: Task Mapping Programming Paradigm for Deep Learning Tensor Programs ASPLOS'23](https://github.com/MachineLearningSystem/hidet) 
  
  paper [Hidet: Task Mapping Programming Paradigm for Deep Learning Tensor Programs ASPLOS'23](https://dl.acm.org/doi/pdf/10.1145/3575693.3575702)

- code [iGniter: Interference-Aware GPU Resource Provisioning for Predictable DNN Inference in the Cloud TPDS'22 ](https://github.com/MachineLearningSystem/igniter) 

   paper [iGniter: Interference-Aware GPU Resource Provisioning for Predictable DNN Inference in the Cloud TPDS'22 ](https://arxiv.org/pdf/2211.01713.pdf)

- code [CheckFreq: Frequent, Fine-Grained DNN Checkpointing FAST'22](https://github.com/MachineLearningSystem/CheckFreq) 

  paper [CheckFreq: Frequent, Fine-Grained DNN Checkpointing FAST'22](https://www.usenix.org/system/files/fast21-mohan.pdf)

- code [Efficient Quantized Sparse Matrix Operations on Tensor Cores  SC'22](https://github.com/MachineLearningSystem/Magicube)

  paper [Efficient Quantized Sparse Matrix Operations on Tensor Cores  SC'22](https://dl.acm.org/doi/pdf/10.5555/3571885.3571934)

- code [Harmony: Overcoming the hurdles of GPU memory capacity to train massive DNN models on commodity servers VLDB'22](https://github.com/MachineLearningSystem/harmony) 

   paper [Harmony: Overcoming the hurdles of GPU memory capacity to train massive DNN models on commodity servers VLDB'22](https://vldb.org/pvldb/vol15/p2747-li.pdf)

- code [PetS: A Unified Framework for Parameter-Efficient Transformers Serving ATC'22  ](https://github.com/MachineLearningSystem/PetS-ATC-2022)

  paper [PetS: A Unified Framework for Parameter-Efficient Transformers Serving ATC'22  ](https://www.usenix.org/system/files/atc22-zhou-zhe.pdf)

- code [PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections OSDI'21](https://github.com/MachineLearningSystem/pet-osdi21-ae) 

  paper [PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections OSDI'21](https://www.usenix.org/system/files/osdi21-wang-haojie.pdf)

- code [APNN-TC: Accelerating Arbitrary Precision Neural Networks on Ampere GPU Tensor Core SC'21](https://github.com/MachineLearningSystem/APNN-TC) 

  paper [APNN-TC: Accelerating Arbitrary Precision Neural Networks on Ampere GPU Tensor Core SC'21](https://dl.acm.org/doi/pdf/10.1145/3458817.3476157)

- code [iGUARD: In-GPU Advanced Race Detection SOSP'21](https://github.com/MachineLearningSystem/iGUARD.git) 

  paper [iGUARD: In-GPU Advanced Race Detection SOSP'21](https://dl.acm.org/doi/pdf/10.1145/3477132.3483545)

- code [Fluid: Resource-Aware Hyperparameter Tuning Engine MLSYS'21](https://github.com/MachineLearningSystem/Fluid) 

  paper [Fluid: Resource-Aware Hyperparameter Tuning Engine MLSYS'21](https://www.mosharaf.com/wp-content/uploads/fluid-mlsys21.pdf)

- code [Baechi: Fast Device Placement on Machine Learning Graphs SOCC'20 ](https://github.com/MachineLearningSystem/baechi)

  paper [Baechi: Fast Device Placement on Machine Learning Graphs SOCC'20 ](https://dprg.cs.uiuc.edu/data/files/2020/socc20-final352.pdf)

- code [Dynamic Parameter Allocation in Parameter Servers VLDB'20 ](https://github.com/MachineLearningSystem/AdaPS) 

  paper [Dynamic Parameter Allocation in Parameter Servers VLDB'20 ](https://www.vldb.org/pvldb/vol13/p1877-renz-wieland.pdf)

- code [Data Movement Is All You Need: A Case Study on Optimizing Transformers](https://github.com/MachineLearningSystem/substation) 
 paper [Data Movement Is All You Need: A Case Study on Optimizing Transformers](https://htor.inf.ethz.ch/publications/img/data_movement_is_all_you_need.pdf)

### GNN
- code [gSampler: Efficient GPU-Based Graph Sampling for Graph Learning SOSP'23](https://github.com/MachineLearningSystem/23SOSP-gSampler)
   
   paper [gSampler: Efficient GPU-Based Graph Sampling for Graph Learning SOSP'23](https://dl.acm.org/doi/pdf/10.1145/3600006.3613168)

- code [Legion: Automatically Pushing the Envelope of Multi-GPU System for Billion-Scale GNN Training ATC'23](https://github.com/MachineLearningSystem/ATC23-Legion)

  paper [Legion: Automatically Pushing the Envelope of Multi-GPU System for Billion-Scale GNN Training ATC'23](https://www.usenix.org/system/files/atc23-sun.pdf)

- code [TC-GNN: Accelerating Sparse Graph Neural Network Computation Via Dense Tensor Core on GPUs ATC'23](https://github.com/MachineLearningSystem/ATC23-TCGNN-Pytorch)

  paper [TC-GNN: Accelerating Sparse Graph Neural Network Computation Via Dense Tensor Core on GPUs ATC'23](https://www.usenix.org/system/files/atc23-wang-yuke.pdf)

- code [Accelerating Graph Neural Networks with Fine-grained intra-kernel Communication-Computation Pipelining on Multi-GPU Platforms OSDI'23](https://github.com/MachineLearningSystem/MGG-OSDI23-AE) 

  paper [Accelerating Graph Neural Networks with Fine-grained intra-kernel Communication-Computation Pipelining on Multi-GPU Platforms OSDI'23](https://www.usenix.org/system/files/osdi23-wang-yuke.pdf)

- code [CoGNN: Efficient Scheduling for Concurrent GNN Training on GPUs SC'22](https://github.com/MachineLearningSystem/CoGNN_info_for_SC22.git) 

  paper [CoGNN: Efficient Scheduling for Concurrent GNN Training on GPUs SC'22](https://dl.acm.org/doi/pdf/10.5555/3571885.3571936)

- code [GNNAdvisor: An Efficient Runtime System for GNN Acceleration on GPUs OSDI'21](https://github.com/MachineLearningSystem/OSDI21_AE-GNN) 
 
  paper  [GNNAdvisor: An Efficient Runtime System for GNN Acceleration on GPUs OSDI'21](https://www.usenix.org/system/files/osdi21-wang-yuke.pdf)

- code [Marius: Learning Massive Graph Embeddings on a Single Machine OSDI'21](https://github.com/MachineLearningSystem/marius) 
  
  paper [Marius: Learning Massive Graph Embeddings on a Single Machine OSDI'21](https://www.usenix.org/system/files/osdi21-mohoney.pdf)

- code [Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads OSDI'21](https://github.com/MachineLearningSystem/dorylus)  
 
  paper [Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads OSDI'21](https://www.usenix.org/system/files/osdi21-thorpe.pdf)

- code [BNS-GCN: Efficient Full-Graph Training of Graph Convolutional Networks with Partition-Parallelism and Random Boundary Node Sampling MLSYS'22 ](https://github.com/MachineLearningSystem/BNS-GCN)

  paper  [BNS-GCN: Efficient Full-Graph Training of Graph Convolutional Networks with Partition-Parallelism and Random Boundary Node Sampling MLSYS'22 ](https://arxiv.org/pdf/2203.10983.pdf)

- code [Accelerating Large Scale Real-Time GNN Inference Using Channel Pruning  VLDB'21 ](https://github.com/MachineLearningSystem/GCNP)

  paper [Accelerating Large Scale Real-Time GNN Inference Using Channel Pruning  VLDB'21 ](http://vldb.org/pvldb/vol14/p1597-zhou.pdf)

- code [Reducing Communication in Graph Neural Network Training SC'20 ](https://github.com/MachineLearningSystem/CAGNET) 

  paper  [Reducing Communication in Graph Neural Network Training SC'20 ](https://dl.acm.org/doi/pdf/10.5555/3433701.3433794)

- [awesome GNN](https://github.com/chwan1016/awesome-gnn-systems)

### Fine-Tune
-  code [Fine-tuning giant neural networks on commodity hardware with automatic pipeline model parallelism ATC'21](https://github.com/MachineLearningSystem/FTPipe-ATC21-Finetune.git) 

   paper [Fine-tuning giant neural networks on commodity hardware with automatic pipeline model parallelism ATC'21](https://www.usenix.org/system/files/atc21-eliad.pdf)

### Energy

- code [Zeus: Understanding and Optimizing {GPU} Energy Consumption of {DNN} Training NSDI'23](https://github.com/MachineLearningSystem/Zeus)

  paper [Zeus: Understanding and Optimizing {GPU} Energy Consumption of {DNN} Training NSDI'23](https://www.usenix.org/system/files/nsdi23-you.pdf)

- code [EnvPipe: Performance-preserving DNN Training Framework for Saving Energy ATC'23](https://github.com/MachineLearningSystem/23ATC-EnvPipe)

  paper  [EnvPipe: Performance-preserving DNN Training Framework for Saving Energy ATC'23](https://www.usenix.org/system/files/atc23-choi.pdf)

### Misc 
- code [Characterizing Variability in Large-Scale, Accelerator-Rich Systems  SC'22 ](https://github.com/MachineLearningSystem/gpu_variability_sc22_artifact)

  paper [Characterizing Variability in Large-Scale, Accelerator-Rich Systems  SC'22 ](https://dl.acm.org/doi/pdf/10.5555/3571885.3571971)

- code [Prediction of the Resource Consumption of Distributed Deep Learning Systems SIGMETRICS'22 ](https://github.com/MachineLearningSystem/Driple) 

  paper [Prediction of the Resource Consumption of Distributed Deep Learning Systems SIGMETRICS'22 ](https://dl.acm.org/doi/pdf/10.1145/3530895)

- code [AI-Enabling Workloads on Large-Scale GPU-Accelerated System: Characterization, Opportunities, and Implications HPCA'22](https://github.com/MachineLearningSystem/HPCA22_SuperCloud)  

  paper [AI-Enabling Workloads on Large-Scale GPU-Accelerated System: Characterization, Opportunities, and Implications HPCA'22](https://baolin-li.netlify.app/uploads/HPCA_2022_MIT_SuperCloud.pdf)



## Contribute
We encourage all contributions to this repository. Open an [issue](https://github.com/lambda7xx/awesome-AI-system/issues) or send a [pull request](https://github.com/lambda7xx/awesome-AI-system/pulls).
