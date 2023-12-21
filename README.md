# Awesome AI System

This repo is motivated by [awesome tensor compilers](https://github.com/merrymercy/awesome-tensor-compilers.git).
## Contents

- [Paper-Code](#paper-code)
  - [LLM Serving Framework](#llm-serving-framework)
  - [LLM Serving](#LLM-Serving)
  - [LLM Platform](#LLM-Platform)
  - [LLM FineTune](#LLM-FineTune)
  - [Fancy LLM](#Fancy-LLM)
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
- code [xFasterTransformer(CPU Side)](https://github.com/intel/xFasterTransformer)
- code [TensorRT-LLM by nvidia ](https://github.com/NVIDIA/TensorRT-LLM.git)
- code [CTranslate2(low latency)](https://github.com/OpenNMT/CTranslate2.git)

### LLM Serving

- code [Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity vldb'24](https://github.com/MachineLearningSystem/24vldb-flash-llm)
  
  paper [Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity vldb'24](https://www.vldb.org/pvldb/vol17/p211-xia.pdf)

- code [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](https://github.com/MachineLearningSystem/PowerInfer.git)
  
  paper [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf)

- code [vLLM System(Efficient Memory Management for Large Language Model Serving with PagedAttention SOSP'23)](https://github.com/vllm-project/vllm)

  paper [Efficient Memory Management for Large Language Model Serving with PagedAttention SOSP'23](https://arxiv.org/pdf/2309.06180.pdf)

- code [SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification 23arxiv](https://github.com/flexflow/FlexFlow/tree/inference)

  paper [SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification 23arxiv](https://arxiv.org/abs/2305.09781.pdf)

### LLM Platform
- code [FastChat](https://github.com/lm-sys/FastChat.git)

### LLM FineTune
- code [S-LoRA: Serving Thousands of Concurrent LoRA Adapters ](https://github.com/S-LoRA/S-LoRA.git)

  paper [S-LoRA: Serving Thousands of Concurrent LoRA Adapters ](https://arxiv.org/pdf/2311.03285.pdf)

- code [Punica: Serving multiple LoRA finetuned LLM as one](https://github.com/punica-ai/punica.git)
 
  paper [Punica: Serving multiple LoRA finetuned LLM as one](https://arxiv.org/pdf/2310.18547.pdf)

### Fancy LLM
- code [LLMCompiler: An LLM Compiler for Parallel Function Calling](https://github.com/MachineLearningSystem/LLMCompiler)

  paper [LLMCompiler: An LLM Compiler for Parallel Function Calling 23arxiv](https://arxiv.org/pdf/2312.04511.pdf)

- code [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://github.com/state-spaces/mamba/tree/main)
  
  paper [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/pdf/2312.00752.pdf)

- code [Teaching LLMs memory management for unbounded context arxiv](https://github.com/MachineLearningSystem/23arxiv-MemGPT)

  paper [MEMGPT: TOWARDS LLMS AS OPERATING SYSTEMS](https://github.com/MachineLearningSystem/23arxiv-MemGPT)

- code [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://github.com/hao-ai-lab/LookaheadDecoding.git)
  
  blog [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)

- code [EAGLE: Lossless Acceleration of LLM Decoding by Feature Extrapolation](https://github.com/SafeAILab/EAGLE.git)

  blog [EAGLE: Lossless Acceleration of LLM Decoding by Feature Extrapolation](https://sites.google.com/view/eagle-llm)

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

- [ModelKeeper: Accelerating DNN Training via Automated Training Warmup](https://github.com/MachineLearningSystem/ModelKeeper) NSDI'23


- [STRONGHOLD: Fast and Affordable Billion-scale Deep Learning Model Training SC'22](https://github.com/MachineLearningSystem/sc22-ae-big_model) 

- [Whale: Efficient Giant Model Training over Heterogeneous {GPUs}](https://github.com/MachineLearningSystem/EasyParallelLibrary) ATC'22

- [GeePS: Scalable Deep Learning on Distributed GPUs with a GPU-Specialized Parameter Server](https://github.com/MachineLearningSystem/geeps) Eurosys'16


### Communication
- [ARK: GPU-driven Code Execution for Distributed Deep Learning NSDI'23](https://github.com/MachineLearningSystem/23NSDI-arkwo)
- [TopoOpt: Optimizing the Network Topology for Distributed DNN Training](https://github.com/MachineLearningSystem/TopoOpt) NSDI'23 

- [Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads](https://github.com/parasailteam/coconet.git) ASPLOS'22 

- [Efficient Sparse Collective Communication and its application to Accelerate Distributed Deep Learning SIGCOMM'21 ](https://github.com/MachineLearningSystem/omnireduce.git) 

### Serving-Inference

- [Paella: Low-latency Model Serving with Virtualized GPU Scheduling SOSP'23](https://github.com/MachineLearningSystem/23sosp-paella)

- [Beta: Statistical Multiplexing with Model Parallelism for Deep Learning Serving OSDI'23](https://github.com/MachineLearningSystem/OSDI23-mms)

- [Optimizing Dynamic Neural Networks with Brainstorm OSDI'23](https://github.com/MachineLearningSystem/23OSDI-brainstorm)

- [Fast and Efficient Model Serving Using Multi-GPUs with Direct-Host-Access](https://github.com/MachineLearningSystem/DeepPlan.git) Eurosys'23

- [Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs.](https://github.com/MachineLearningSystem/hidet)

- [MPCFormer: fast, performant, and private transformer inference with MPC](https://github.com/DachengLi1/MPCFormer) ICLR'23 

- [High-throughput Generative Inference of Large Language Modelwith a Single GPU](https://github.com/MachineLearningSystem/FlexGen) 

- [VELTAIR: Towards High-Performance Multi-Tenant Deep Learning Serving via Adaptive Compilation and Scheduling](https://github.com/MachineLearningSystem/VELTAIR_ASPLOS22) ASPLOS'22

- [DVABatch: Diversity-aware Multi-Entry Multi-Exit Batching for Efficient Processing of DNN Services on GPUs](https://github.com/MachineLearningSystem/DVABatch)  ATC'22 

- [Cocktail: A Multidimensional Optimization for Model Serving in Cloud](https://github.com/MachineLearningSystem/cocktail) NSDI'22

- [Serving Heterogeneous Machine Learning Models on Multi-GPU Servers with Spatio-Temporal Sharing](https://github.com/MachineLearningSystem/glet) ATC'22

- [RIBBON: cost-effective and qos-aware deep learning model inference using a diverse pool of cloud computing instances](https://github.com/MachineLearningSystem/SC21_Ribbon) SC'21

- [ INFaaS: Automated Model-less Inference Serving](https://github.com/MachineLearningSystem/INFaaS.git) ATC'21

- [Abacus](https://github.com/MachineLearningSystem/Abacus) SC'21

- [Serving DNNs like Clockwork: Performance Predictability from the Bottom Up](https://github.com/MachineLearningSystem/clockwork) OSDI'20

- [Exploiting Cloud Services for Cost-Effective, SLO-Aware Machine Learning Inference Serving](https://github.com/MachineLearningSystem/MArk-Project) ATC'19 

- [Nexus: a GPU cluster engine for accelerating DNN-based video analysis](https://github.com/MachineLearningSystem/nexus) SOSP'19 

- [Clipper:A low-latency prediction-serving system](https://github.com/ucbrise/clipper) NSDI'17


### MoE
- [SmartMoE: Efficiently Training Sparsely-Activated Models through Combining Static and Dynamic Parallelization ATC'23](https://github.com/MachineLearningSystem/23ATC-SmartMoE-AE)
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://github.com/stanford-futuredata/megablocks) MLSYS'23 
- [Tutel: Adaptive Mixture-of-Experts at Scale](https://github.com/MachineLearningSystem/tutel-MOE) MLSYS'23

- [FastMoE: A Fast Mixture-of-Expert Training System](https://github.com/MachineLearningSystem/fastmoe-thu) PPOPP'22

- [awesome MoE](https://github.com/MachineLearningSystem/awesome-mixture-of-experts)

- [MoE Paper](https://github.com/MachineLearningSystem/Awesome-Mixture-of-Experts-Papers)

- [AutoMoE: Neural Architecture Search for Efficient Sparsely Activated Transformers](https://github.com/MachineLearningSystem/AutoMoE) 

### GPU Cluster Management
- [Lucid: A Non-Intrusive, Scalable and Interpretable Scheduler for Deep Learning Training Jobs](https://github.com/MachineLearningSystem/Lucid) ASPLOS'23

- [Shockwave: Fair and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning](https://github.com/MachineLearningSystem/shockwave) NSDI'23

- [Synergy : Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters](https://github.com/MachineLearningSystem/synergy.git) OSDI'22

- [Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning](https://github.com/MachineLearningSystem/adaptdl) OSDI'21

- [Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads](https://github.com/MachineLearningSystem/gavel) OSDI'20

- [Tiresias -- A GPU Cluster Manager for Distributed Deep Learning Training without complete job information](https://github.com/MachineLearningSystem/Tiresias) NSDI'19

- [Chronus: A Novel Deadline-aware Scheduler for Deep Learning Training Jobs](https://github.com/MachineLearningSystem/ChronusArtifact) SOCC'21 


- [awesome DL scheduler](https://github.com/MachineLearningSystem/Awesome-DL-Scheduling-Papers.git)

### Schedule and Resource Management
- [An interference-aware scheduler for fine-grained GPU sharing Resources Eurosys'24](https://github.com/MachineLearningSystem/24Eurosys-orion.git)
- [ElasticFlow: An Elastic Serverless Training Platform for Distributed Deep Learning](https://github.com/MachineLearningSystem/ElasticFlow-ASPLOS23) ASPLOS'23 

- [Multi-Resource Interleaving for Deep Learning Training](https://github.com/MachineLearningSystem/Muri) SIGCOMM'22

- [Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training ](https://github.com/MachineLearningSystem/slapo)  arxiv 

- [Out-of-order backprop: an effective scheduling technique for deep learning](https://github.com/MachineLearningSystem/ooo-backprop) Eurosys'22 

- [ KungFu: Making Training in Distributed Machine Learning Adaptive](https://github.com/MachineLearningSystem/KungFu) OSDI'20

- [PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications](https://github.com/MachineLearningSystem/PipeSwitch) OSDI'20 


### Optimization
- [GLake: optimizing GPU memory management and IO transmission ASPLOS'24](https://github.com/MachineLearningSystem/24ASPLOS-glake)
- [Spada: Accelerating Sparse Matrix Multiplication with Adaptive Dataflow](https://github.com/MachineLearningSystem/spada-sim) ASPLOS'23 

- [MISO: Exploiting Multi-Instance GPU Capability on Multi-Tenant GPU Clusters](https://github.com/MachineLearningSystem/socc22-miso) SOCC'22 

- [Accpar: Tensor partitioning for heterogeneous deep learning accelerators](https://github.com/MachineLearningSystem/AccPar) HPCA'20 


- [Hidet: Task Mapping Programming Paradigm for Deep Learning Tensor Programs](https://github.com/MachineLearningSystem/hidet) ASPLOS'23

- [iGniter: Interference-Aware GPU Resource Provisioning for Predictable DNN Inference in the Cloud](https://github.com/MachineLearningSystem/igniter) TPDS'22 

- [CheckFreq: Frequent, Fine-Grained DNN Checkpointing](https://github.com/MachineLearningSystem/CheckFreq) FAST'22

- [Efficient Quantized Sparse Matrix Operations on Tensor Cores](https://github.com/MachineLearningSystem/Magicube) SC'22

- [Harmony: Overcoming the hurdles of GPU memory capacity to train massive DNN models on commodity servers](https://github.com/MachineLearningSystem/harmony) VLDB'22

- [Pets](https://github.com/MachineLearningSystem/PetS-ATC-2022) ATC'22 

- [PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections](https://github.com/MachineLearningSystem/pet-osdi21-ae) OSDI'21

- [APNN-TC: Accelerating Arbitrary Precision Neural Networks on Ampere GPU Tensor Cores](https://github.com/MachineLearningSystem/APNN-TC) SC'21

- [iGUARD](https://github.com/MachineLearningSystem/iGUARD.git) SOSP'21

- [Fluid: Resource-Aware Hyperparameter Tuning Engine](https://github.com/MachineLearningSystem/Fluid) MLSYS'21
- [Baechi: Fast Device Placement on Machine Learning Graphs ](https://github.com/MachineLearningSystem/baechi) SOCC'20 

- [Dynamic Parameter Allocation in Parameter Servers](https://github.com/MachineLearningSystem/AdaPS) VLDB'20 

- [Data Movement Is All You Need: A Case Study on Optimizing Transformers](https://github.com/MachineLearningSystem/substation) 

### GNN
- [gSampler: Efficient GPU-Based Graph Sampling for Graph Learning SOSP'23](https://github.com/MachineLearningSystem/23SOSP-gSampler)
- [Legion: Automatically Pushing the Envelope of Multi-GPU System for Billion-Scale GNN Training ATC'23](https://github.com/MachineLearningSystem/ATC23-Legion)
- [TC-GNN: Accelerating Sparse Graph Neural Network Computation Via Dense Tensor Core on GPUs ATC'23](https://github.com/MachineLearningSystem/ATC23-TCGNN-Pytorch)
- [Accelerating Graph Neural Networks with Fine-grained intra-kernel Communication-Computation Pipelining on Multi-GPU Platforms](https://github.com/MachineLearningSystem/MGG-OSDI23-AE) OSDI'23
- [COGNN](https://github.com/MachineLearningSystem/CoGNN_info_for_SC22.git) SC'22
- [TC-GNN: Accelerating Sparse Graph Neural Network Computation Via Dense Tensor Core on GPUs](https://github.com/MachineLearningSystem/TCGNN-Pytorch)
- [GNNAdvisor: An Efficient Runtime System for GNN Acceleration on GPUs](https://github.com/MachineLearningSystem/OSDI21_AE-GNN) OSDI'21

- [Marius: Learning Massive Graph Embeddings on a Single Machine](https://github.com/MachineLearningSystem/marius) OSDI'21

- [Dorylus: Affordable, Scalable, and Accurate GNN Training with Distributed CPU Servers and Serverless Threads](https://github.com/MachineLearningSystem/dorylus) OSDI'21 

- [BNS-GCN: Efficient Full-Graph Training of Graph Convolutional Networks with Partition-Parallelism and Random Boundary Node Sampling](https://github.com/MachineLearningSystem/BNS-GCN) MLSYS'22 

- [Accelerating Large Scale Real-Time GNN Inference Using Channel Pruning](https://github.com/MachineLearningSystem/GCNP) VLDB'21 
- [Reducing Communication in Graph Neural Network Training](https://github.com/MachineLearningSystem/CAGNET) SC'20 

- [awesome GNN](https://github.com/chwan1016/awesome-gnn-systems)

### Fine-Tune
-  [Fine-tuning giant neural networks on commodity hardware with automatic pipeline model parallelism](https://github.com/MachineLearningSystem/FTPipe-ATC21-Finetune.git) ATC'21

### Energy

- [Zeus: Understanding and Optimizing {GPU} Energy Consumption of {DNN} Training NSDI'23](https://github.com/MachineLearningSystem/Zeus) 

- [EnvPipe: Performance-preserving DNN Training Framework for Saving Energy ATC'23](https://github.com/MachineLearningSystem/23ATC-EnvPipe)

### Misc 
- [Characterizing Variability in Large-Scale, Accelerator-Rich Systems](https://github.com/MachineLearningSystem/gpu_variability_sc22_artifact) SC'22 

- [Prediction of the Resource Consumption of Distributed Deep Learning Systems](https://github.com/MachineLearningSystem/Driple) SIGMETRICS'22 

- [AI-Enabling Workloads on Large-Scale GPU-Accelerated System: Characterization, Opportunities, and Implications](https://github.com/MachineLearningSystem/HPCA22_SuperCloud) HPCA'22



## Contribute
We encourage all contributions to this repository. Open an [issue](https://github.com/lambda7xx/awesome-AI-system/issues) or send a [pull request](https://github.com/lambda7xx/awesome-AI-system/pulls).
