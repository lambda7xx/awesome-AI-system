# Awesome AI System

This repo is motivated by [awesome tensor compilers](https://github.com/merrymercy/awesome-tensor-compilers.git).
## Contents

- [Paper-Code](#paper-code)
  - [Researcher](#researcher)
  - [LLM Serving Framework](#llm-serving-framework)
  - [LLM Evaluation Platform](#llm-evaluation-platform)
  - [LLM Robustness and Debugging](#llm-robustness-and-debugging)
  - [LLM Inference System Side)](#llm-inference-system-side)
  - [Compiler](#compiler)
  - [Attention](#attention)
  - [RAG And ANNS](#rag-and-anns)
  - [RLHF](#rlhf)
  - [Video](#video)
  - [LLM Inference AI Side)](#llm-inference-ai-side)
  - [LLM MoE](#llm-moe)
  - [LoRA](#lora)
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

### Researcher 



| Name | University | Homepage | 
|:-----:|:-----:|:-----:|
| Ion Stoica | UC Berkeley | [![Website](https://img.shields.io/badge/Website-9cf)](https://people.eecs.berkeley.edu/~istoica/) |
| Joseph E. Gonzalez | UC Berkeley | [![Website](https://img.shields.io/badge/Website-9cf)](https://people.eecs.berkeley.edu/~jegonzal/) |
| Matei Zaharia | UC Berkeley | [![Website](https://img.shields.io/badge/Website-9cf)](https://people.eecs.berkeley.edu/~matei/) |
| Zhihao Jia| CMU | [![Website](https://img.shields.io/badge/Website-9cf)](https://www.cs.cmu.edu/~zhihaoj2/) |
| Tianqi Chen| CMU | [![Website](https://img.shields.io/badge/Website-9cf)](https://tqchen.com/) |
| Stephanie Wang | UW | [![Website](https://img.shields.io/badge/Website-9cf)](https://stephanie-wang.github.io/) |
| Xingda Wei| SJTU | [![Website](https://img.shields.io/badge/Website-9cf)](https://ipads.se.sjtu.edu.cn/pub/members/xingda_wei) |
| Zeyu Min| SJTU | [![Website](https://img.shields.io/badge/Website-9cf)](https://ipads.se.sjtu.edu.cn/pub/members/zeyu_mi) |
| Xin Jin | PKU | [![Website](https://img.shields.io/badge/Website-9cf)](https://xinjin.github.io/) |
| Harry Xu | UCLA | [![Website](https://img.shields.io/badge/Website-9cf)](https://web.cs.ucla.edu/~harryxu/) |
| Anand Iyer | Georgia Tech | [![Website](https://img.shields.io/badge/Website-9cf)](https://www.anand-iyer.com/) |
| Ravi Netravali| Princeton | [![Website](https://img.shields.io/badge/Website-9cf)](https://www.cs.princeton.edu/~ravian/) |
| Christos Kozyrakis | Stanford | [![Website](https://img.shields.io/badge/Website-9cf)](https://web.stanford.edu/~kozyraki/) |
| Christopher Ré | Stanford | [![Website](https://img.shields.io/badge/Website-9cf)](https://cs.stanford.edu/people/chrismre/) |
| Tri Dao| Princeton | [![Website](https://img.shields.io/badge/Website-9cf)](https://tridao.me/) |
| Mosharaf Chowdhury| UMich | [![Website](https://img.shields.io/badge/Website-9cf)](https://www.mosharaf.com/) |
| Shivaram Venkataraman| Wisc | [![Website](https://img.shields.io/badge/Website-9cf)](https://shivaram.org/) |
| Hao Zhang| UCSD | [![Website](https://img.shields.io/badge/Website-9cf)](https://cseweb.ucsd.edu/~haozhang/) |
| Yiying Zhang| UCSD | [![Website](https://img.shields.io/badge/Website-9cf)](https://cseweb.ucsd.edu/~yiying/) |
| Ana Klimovic | ETH | [![Website](https://img.shields.io/badge/Website-9cf)](https://anakli.inf.ethz.ch/) |
| Fan Lai | UIUC | [![Website](https://img.shields.io/badge/Website-9cf)](https://www.fanlai.me/) |
| Lianmin Zheng | UC Berkeley | [![Website](https://img.shields.io/badge/Website-9cf)](https://lmzheng.net/) |
| Ying Sheng  | Stanford | [![Website](https://img.shields.io/badge/Website-9cf)](https://sites.google.com/view/yingsheng/) |
| Zhuohan Li | UC Berkeley | [![Website](https://img.shields.io/badge/Website-9cf)](https://people.eecs.berkeley.edu/~zhuohan/) |
| Woosuk Kwon| UC Berkeley | [![Website](https://img.shields.io/badge/Website-9cf)](https://woosuk.me/) |
| Zihao Ye | University of Washington  | [![Website](https://img.shields.io/badge/Website-9cf)](https://homes.cs.washington.edu/~zhye/) |
| Amey Agrawal | Georgia Tech | [![Website](https://img.shields.io/badge/Website-9cf)](https://ameya.info/) |


### LLM Serving Framework

| Title | Github|
|:-----:|:-----:|
| MLC LLM| [![Star](https://img.shields.io/github/stars/mlc-ai/mlc-llm.svg)](https://github.com/mlc-ai/mlc-llm/) |
| TensorRT-LLM | [![Star](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg)](https://github.com/NVIDIA/TensorRT-LLM.git) |
| xFasterTransformer |  [![Star](https://img.shields.io/github/stars/intel/xFasterTransformer.svg)](https://github.com/intel/xFasterTransformer)|
| CTranslate2(low latency) | [![Star](https://img.shields.io/github/stars/OpenNMT/CTranslate2.svg)](https://github.com/OpenNMT/CTranslate2.git)|
| llama2.c| [![Star](https://img.shields.io/github/stars/karpathy/llama2.c.svg)](https://github.com/karpathy/llama2.c) |


### LLM Evaluation Platform

| Title | Github| Website
|:-----:|:-----:|:-----:|
| FastChat | [![Star](https://img.shields.io/github/stars/lm-sys/FastChat.svg)](https://github.com/lm-sys/FastChat.git)| [![Website](https://img.shields.io/badge/Website-9cf)](https://chat.lmsys.org/) |

### LLM Robustness and Debugging

| Title | Paper | Github | Pub. & Date |
|:-----:|:-----:|:------:|:-----------:|
| WFGY 1.0: Self-healing LLM Systems Framework | [![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.30338884-9cf)](https://doi.org/10.6084/m9.figshare.30338884) <br> [PDF](https://github.com/onestardao/WFGY/blob/main/I_am_not_lizardman/WFGY_All_Principles_Return_to_One_v1.0_PSBigBig_Public.pdf) | [![Star](https://img.shields.io/github/stars/onestardao/WFGY.svg)](https://github.com/onestardao/WFGY) | Tech report, Oct 13 2025 |


### LLM Inference (System Side)
| Title | Paper | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| Aegaeon: Effective GPU Pooling for Concurrent LLM Serving on the Market | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf) |  | - | SOSP'25|
| DiffKV: Differentiated Memory Management for Large Language Models with Parallel KV Compaction | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://dl.acm.org/doi/pdf/10.1145/3731569.3764810) | [![Star](https://img.shields.io/github/stars/zyqCSL/DiffKV.svg)](https://github.com/zyqCSL/DiffKV.git) | - | SOSP'25|
| Pie: A Programmable Serving System for Emerging LLM Applications | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://pie-project.org/assets/files/gim2025pie-205fb6aa1c1b3c9e172dd1db182db8e5.pdf) | [![Star](https://img.shields.io/github/stars/pie-project/pie.svg)](https://github.com/pie-project/pie.git) | - | SOSP'25|
| KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) | [![Star](https://img.shields.io/github/stars/kvcache-ai/ktransformers.svg)](https://github.com/kvcache-ai/ktransformers) | - | SOSP'25|
| XSched: Preemptive Scheduling for Diverse XPUs| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.usenix.org/system/files/osdi25-shen-weihang.pdf) | [![Star](https://img.shields.io/github/stars/XpuOS/xsched.svg)](https://github.com/XpuOS/xsched.git) | - | OSDI 25|
| TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.11329) | [![Star](https://img.shields.io/github/stars/microsoft/tokenweave.svg)](https://github.com/microsoft/tokenweave.git) | - | Arxiv 25|
|  ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.09999) | [![Star](https://img.shields.io/github/stars/alibaba/ServeGen.svg)](https://github.com/alibaba/ServeGen.git) | - | Arxiv 25|
|  Resource Multiplexing in Tuning and Serving Large Language Models | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.usenix.org/system/files/atc25-he-yongjun.pdf) | [![Star](https://img.shields.io/github/stars/llm-db/llmstation.svg)](https://github.com/llm-db/llmstation) | - | ATC'25|
|  RetroInfer: A Vector-Storage Approach for Scalable Long-Context LLM Inference | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.02922) | [![Star](https://img.shields.io/github/stars/microsoft/RetrievalAttention.svg)](https://github.com/microsoft/RetrievalAttention.git) | - | Arxiv May 2025 |
|  SpecEE: Accelerating Large Language Model Inference with Speculative Early Exiting | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2504.08850) | [![Star](https://img.shields.io/github/stars/infinigence/SpecEE.svg)](https://github.com/infinigence/SpecEE) | - | ISCA'25 |
|   LIA: A Single-GPU LLM Inference Acceleration with Cooperative AMX-Enabled CPU-GPU Computation and CXL Offloading | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://dl.acm.org/doi/pdf/10.1145/3695053.3731092) | [![Star](https://img.shields.io/github/stars/hyungyokim/LIA_AMXGPU.svg)](https://github.com/hyungyokim/LIA_AMXGPU) | - | ISCA'25 |
|  Apt-Serve: Adaptive Request Scheduling on Hybrid Cache for Scalable LLM Inference Serving | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2504.07494) | [![Star](https://img.shields.io/github/stars/eddiegaoo/Apt-Serve.svg)](https://github.com/eddiegaoo/Apt-Serve) | - | SIGMOD'25 |
|  Marconi: Prefix Caching for the Era of Hybrid LLMs | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2411.19379) | [![Star](https://img.shields.io/github/stars/ruipeterpan/marconi.svg)](https://github.com/ruipeterpan/marconi) | - | MLSys'25 |
|  SpInfer: Leveraging Low-Level Sparsity for Efficient Large Language Model Inference on GPUs | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://dl.acm.org/doi/10.1145/3689031.3717481) | [![Star](https://img.shields.io/github/stars/MachineLearningSystem/25Eurosys-SpInfer.svg)](https://github.com/MachineLearningSystem/25Eurosys-SpInfer) | - | Eurosys'25 Best Paper |
|  NeuStream: Bridging Deep Learning Serving and Stream Processing | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://dl.acm.org/doi/10.1145/3689031.3717489) | [![Star](https://img.shields.io/github/stars/MachineLearningSystem/25Eurosys-NeuStream-AE.svg)](https://github.com/MachineLearningSystem/25Eurosys-NeuStream-AE) | - | Eurosys'25 |
|  Towards End-to-End Optimization of LLM-based Applications with Ayo | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2407.00326) | [![Star](https://img.shields.io/github/stars/MachineLearningSystem/25ASPLOS-Ayo.svg)](https://github.com/MachineLearningSystem/25ASPLOS-Ayo) | - | ASPLOS'25 |
|  NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.01142) | [![Star](https://img.shields.io/github/stars/MachineLearningSystem/25MLSYS-NEO.svg)](https://github.com/MachineLearningSystem/25MLSYS-NEO) | - | MLSYS'25 |
| CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.16444) | [![Star](https://img.shields.io/github/stars/YaoJiayi/CacheBlend.svg)](https://github.com/YaoJiayi/CacheBlend) | - | Eurosys'25 Best Paper|
| Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.01566) | [![Star](https://img.shields.io/github/stars/Thesys-lab/Helix-ASPLOS25.svg)](https://github.com/Thesys-lab/Helix-ASPLOS25.git) | - | ASPLOS'25 |
|GLINTHAWK: A Two-Tiered Architecture for High-Throughput LLM Inference  | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2501.11779) | [![Star](https://img.shields.io/github/stars/microsoft/glinthawk.svg)](https://github.com/microsoft/glinthawk) | - | Arxiv'25,Jan |
| Queue Management for SLO-Oriented Large Language Model Serving | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://haoran-qiu.com/pdf/socc24-qlm.pdf) | [![Star](https://img.shields.io/github/stars/QLM-project/QLM.svg)](https://github.com/QLM-project/QLM.git) | - | SOCC'24 |
|NanoFlow: Towards Optimal Large Language Model Serving Throughput | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.12757) | [![Star](https://img.shields.io/github/stars/efeslab/Nanoflow.svg)](https://github.com/efeslab/Nanoflow.git) | - | OSDI'25 |
| PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf) | [![Star](https://img.shields.io/github/stars/SJTU-IPADS/PowerInfer.svg)](https://github.com/SJTU-IPADS/PowerInfer) | - | SOSP'24 |
|LoongServe: Efficiently Serving Long-context Large Language Models with Elastic Sequence Parallelism | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2404.09526) | [![Star](https://img.shields.io/github/stars/LoongServe/LoongServe.svg)](https://github.com/LoongServe/LoongServe) | - | SOSP'24 |
|Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.09054) | [![Star](https://img.shields.io/github/stars/d-matrix-ai/keyformer-llm.svg)](https://github.com/d-matrix-ai/keyformer-llm) | - | MLSYS'24 |
|PLLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2308.12066) | [![Star](https://img.shields.io/github/stars/PrincetonUniversity/LLMCompass.svg)](https://github.com/PrincetonUniversity/LLMCompass) | - | ISCA'24 |
|Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2308.12066) | [![Star](https://img.shields.io/github/stars/ranggihwang/Pregated_MoE.svg)](https://github.com/ranggihwang/Pregated_MoE) | - | ISCA'24 |
|Prompt Cache: Modular Attention Reuse for Low-Latency Inference| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2311.04934) | [![Star](https://img.shields.io/github/stars/yale-sys/prompt-cache.svg)](https://github.com/yale-sys/prompt-cache) | - | MLSYS'24 |
|Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2403.02310) | [![Star](https://img.shields.io/github/stars/microsoft/sarathi-serve.svg)](https://github.com/microsoft/sarathi-serve) | - | OSDI'24 |
| DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.09670) | [![Star](https://img.shields.io/github/stars/LLMServe/DistServe.svg)](https://github.com/LLMServe/DistServe) | - | OSDI'24 |
| Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.00079) | [![Star](https://img.shields.io/github/stars/kvcache-ai/Mooncake.svg)](https://github.com/kvcache-ai/Mooncake.git) | - | July'24 |
|Llumnix: Dynamic Scheduling for Large Language Model Serving | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.03243) | [![Star](https://img.shields.io/github/stars/AlibabaPAI/llumnix.svg)](https://github.com/AlibabaPAI/llumnix/tree/osdi24ae) | - | OSDI'24 |
| Parrot: Efficient Serving of LLM-based Application with Semantic Variables| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2310.07240) | [![Star](https://img.shields.io/github/stars/MachineLearningSystem/24OSDI-ParrotServe.svg)](https://github.com/MachineLearningSystem/24OSDI-ParrotServe) | - | OSDI'24 |
| CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2310.07240) | [![Star](https://img.shields.io/github/stars/UChi-JCL/CacheGen.svg)](https://github.com/UChi-JCL/CacheGen) | - | SIGCOMM'24 |
| Efficiently Programming Large Language Models using SGLang| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2312.07104) | [![Star](https://img.shields.io/github/stars/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang.git) | - | Jan, 2024 |
| Efficient Memory Management for Large Language Model Serving with PagedAttention| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2309.06180.pdf) | [![Star](https://img.shields.io/github/stars/vllm-project/vllm.svg)](https://github.com/vllm-project/vllm.git) | - | SOSP'23 |
| SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.09781.pdf) | [![Star](https://img.shields.io/github/stars/flexflow/FlexFlow.svg)](https://github.com/flexflow/FlexFlow) | - | Dec,2023 |
|Liger: Interleaving Intra- and Inter-Operator Parallelism for Distributed Large Model Inference| - | [![Star](https://img.shields.io/github/stars/MachineLearningSystem/24PPOPP-Liger.svg)](https://github.com/MachineLearningSystem/24PPOPP-Liger) |-| PPOPP'24
|Efficiently Programming Large Language Models using SGLang| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2312.07104.pdf)| [![Star](https://img.shields.io/github/stars/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang.git) | - | Nurips'24 | 
| Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.vldb.org/pvldb/vol17/p211-xia.pdf) | [![Star](https://img.shields.io/github/stars/AlibabaResearch/flash-llm.svg)](https://github.com/AlibabaResearch/flash-llm) | - | VLDB'24 |

### Compiler
| Title | Paper | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| Mercury: Unlocking Multi-GPU Operator Optimization for LLMs via Remote Memory Scheduling| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://dl.acm.org/doi/pdf/10.1145/3731569.3764798) | [![Star](https://img.shields.io/github/stars/ChandlerGuan/mercury_artifact.svg)](https://github.com/ChandlerGuan/mercury_artifact.git) | - | SOSP'25|
| Mirage: A Multi-Level Superoptimizer  for Tensor Programs| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.usenix.org/system/files/osdi25-wu-mengdi.pdf) | [![Star](https://img.shields.io/github/stars/mirage-project/mirage.svg)](https://github.com/mirage-project/mirage.git) | - | OSDI'25|

### Attention
| Title | Paper | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://dl.acm.org/doi/pdf/10.1145/3712285.3759894) | [![Star](https://img.shields.io/github/stars/oliverYoung2001/UltraAttn.svg)](https://github.com/oliverYoung2001/UltraAttn.git) | - | SC'25|
| TASP: Topology-aware Sequence Parallelism | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2509.26541) | [![Star](https://img.shields.io/github/stars/infinigence/HamiltonAttention.svg)](https://github.com/infinigence/HamiltonAttention.git) | - | Arxiv'25|
| Ring Attn | | [![Star](https://img.shields.io/github/stars/zhuzilin/ring-flash-attention.svg)](https://github.com/zhuzilin/ring-flash-attention.git) | - | |



### RAG And ANNS
| Title | Paper | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| HedraRAG: Co-Optimizing Generation and Retrieval for Heterogeneous RAG Workflows| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://dl.acm.org/doi/pdf/10.1145/3731569.3764806) | [![Star](https://img.shields.io/github/stars/Leo9660/HedraRAG_AE.svg)](https://github.com/Leo9660/HedraRAG_AE.git) | - | SOSP'25|
| LEANN: A Low-Storage Vector Index | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)]({https://arxiv.org/abs/2506.08276) | [![Star](https://img.shields.io/github/stars/yichuan-w/LEANN.svg)](https://github.com/yichuan-w/LEANN.git) | - | Arxiv 25 |
| OdinANN: Direct Insert for Consistently Stable Performance in Billion-Scale Graph-Based Vector Search | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.usenix.org/system/files/osdi25-guo.pdf) | [![Star](https://img.shields.io/github/stars/thustorage/PipeANN.svg)](https://github.com/thustorage/PipeANN) | - | FAST'26 |
| Achieving Low-Latency Graph-Based Vector Search via Aligning Best-First Search Algorithm with SSD | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.usenix.org/system/files/osdi25-guo.pdf) | [![Star](https://img.shields.io/github/stars/thustorage/PipeANN.svg)](https://github.com/thustorage/PipeANN) | - | OSDI'25 |
| Quake: Adaptive Indexing for Vector Search| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.usenix.org/system/files/osdi25-mohoney.pdf) | [![Star](https://img.shields.io/github/stars/marius-team/quake.svg)](https://github.com/marius-team/quake) | - | OSDI'25 |
| Hermes: Algorithm-System Co-design for Efficient Retrieval Augmented Generation At-Scale | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://dl.acm.org/doi/pdf/10.1145/3695053.3731076) | [![Star](https://img.shields.io/github/stars/S4AI-CornellTech/Hermes.svg)](https://github.com/S4AI-CornellTech/Hermes) | - | ISCA'25 |
| PathWeaver: A High-Throughput Multi-GPU System for Graph-Based Approximate Nearest Neighbor Search | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.usenix.org/system/files/atc25-kim.pdf) | [![Star](https://img.shields.io/github/stars/AIS-SNU/PathWeaver.svg)](https://github.com/AIS-SNU/PathWeaver.git) | - | ATC'25 |
| In-Storage Acceleration of Retrieval Augmented Generation as a Service: Artifact Evaluation README | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://dl.acm.org/doi/pdf/10.1145/3695053.3731032) | [![Star](https://img.shields.io/github/stars/he-actlab/ragx.svg)](https://github.com/he-actlab/ragx) | - | ISCA'25 |
| RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.14649) | [![Star](https://img.shields.io/github/stars/google/rago.svg)](https://github.com/google/rago.git) | - | ISCA'25 |

### RLHF
| Title | Paper | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| Optimizing RLHF Training for Large Language Models with Stage Fusion | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.13221) | [![Star](https://img.shields.io/github/stars/FlexFusion/FlexFusion.svg)](https://github.com/FlexFusion/FlexFusion.git) | - | NSDI'25 |
| HybridFlow: A Flexible and Efficient RLHF Framework | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2409.19256v2) | [![Star](https://img.shields.io/github/stars/volcengine/verl.svg)](https://github.com/volcengine/verl.git) | - | Eurosys'25 |
| ReaLHF: Optimized RLHF Training for Large Language Models through Parameter Reallocation | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.14088) | [![Star](https://img.shields.io/github/stars/openpsi-project/ReaLHF.svg)](https://github.com/openpsi-project/ReaLHF.git)| - | June. 2024 |
| OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2405.11143) | [![Star](https://img.shields.io/github/stars/OpenRLHF/OpenRLHF.svg)](https://github.com/OpenRLHF/OpenRLHF)| - | May. 2024 |

### Video
| Title | Paper | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| Katz: Efficient Workflow Serving for Diffusion Models with Many Adapters | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.usenix.org/system/files/atc25-li-suyi-katz.pdf) | [![Star](https://img.shields.io/github/stars/modelscope/Katz.svg)](https://github.com/modelscope/Katz) | - | ATC'25 |
| PPipe: Efficient Video Analytics Serving on Heterogeneous GPU Clusters via Pool-Based Pipeline Parallelism | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.usenix.org/system/files/atc25-kong.pdf) | [![Star](https://img.shields.io/github/stars/JonnyKong/PPipe.svg)](https://github.com/JonnyKong/PPipe.git) | - | Nov. 2024 |
| xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.01738) | [![Star](https://img.shields.io/github/stars/xdit-project/xDiT.svg)](https://github.com/xdit-project/xDiT.git) | - | Nov. 2024 |
| FastVideo | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2411.01738) | [![Star](https://img.shields.io/github/stars/hao-ai-lab/FastVideo.svg)](https://github.com/hao-ai-lab/FastVideo.git) | - | Dec. 2024 |




### LLM Inference(AI Side)
| Title | Paper | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| InferCept: Efficient Intercept Support for Augmented Large Language Model Inference | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.01869) | [![Star](https://img.shields.io/github/stars/WukLab/InferCept.svg)](https://github.com/WukLab/InferCept) | - | ICML'24 |
| Online Speculative Decoding| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2310.07177) | [![Star](https://img.shields.io/github/stars/LiuXiaoxuanPKU/OSD.svg)](https://github.com/LiuXiaoxuanPKU/OSD) | - | ICML'24 |
| MuxServe: Flexible Spatial-Temporal Multiplexing for LLM Serving| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2404.02015) | [![Star](https://img.shields.io/github/stars/EfficientLLMSys/MuxServe.svg)](https://github.com/EfficientLLMSys/MuxServe) | - | ICML'24 |
| BitDelta: Your Fine-Tune May Only Be Worth One Bit| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.10193) | [![Star](https://img.shields.io/github/stars/FasterDecoding/BitDelta.svg)](https://github.com/FasterDecoding/BitDelta.git) | - | Feb,2024 |
| Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2401.10774) | [![Star](https://img.shields.io/github/stars/FasterDecoding/Medusa.svg)](https://github.com/FasterDecoding/Medusa.git) | - | Jan,2024 |
| LLMCompiler: An LLM Compiler for Parallel Function Calling| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2312.04511.pdf) | [![Star](https://img.shields.io/github/stars/SqueezeAILab/LLMCompiler.svg)](https://github.com/SqueezeAILab/LLMCompiler.git) | - | Dec,2023 |
| Mamba: Linear-Time Sequence Modeling with Selective State Spaces| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2312.00752.pdf) | [![Star](https://img.shields.io/github/stars/state-spaces/mamba.svg)](https://github.com/state-spaces/mamba.git) | - | Dec,2023 |
| Teaching LLMs memory management for unbounded context| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.08560) | [![Star](https://img.shields.io/github/stars/cpacker/MemGPT.svg)](https://github.com/cpacker/MemGPT.git) | - | Oct,2023 |
| Break the Sequential Dependency of LLM Inference Using Lookahead Decoding| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2402.02057) | [![Star](https://img.shields.io/github/stars/hao-ai-lab/LookaheadDecoding.svg)](https://github.com/hao-ai-lab/LookaheadDecoding.git) | - | Feb,2024 |
| EAGLE: Lossless Acceleration of LLM Decoding by Feature Extrapolation| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2401.15077.pdf) | [![Star](https://img.shields.io/github/stars/SafeAILab/EAGLE.svg)](https://github.com/SafeAILab/EAGLE.git) | - | Jan,2024 |

### LLM MoE
| Title | Paper | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2308.12066) | [![Star](https://img.shields.io/github/stars/ranggihwang/Pregated_MoE.svg)](https://github.com/ranggihwang/Pregated_MoE.git) | - | ISCA'24 |
| SIDA-MOE: SPARSITY-INSPIRED DATA-AWARE SERVING FOR EFFICIENT AND SCALABLE LARGE MIXTURE-OF-EXPERTS MODELS| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2310.18859) | [![Star](https://img.shields.io/github/stars/timlee0212/SiDA-MoE.svg)](https://github.com/timlee0212/SiDA-MoE) | - | MLSYS'24 |
| ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://dl.acm.org/doi/10.1145/3627703.3650083) | [![Star](https://img.shields.io/github/stars/Fragile-azalea/ScheMoE.svg)](https://github.com/Fragile-azalea/ScheMoE.git) | - | Eurosys'24 |


### LoRA

| Title | Paper | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
| dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang) | [![Star](https://img.shields.io/github/stars/LLMServe/dLoRA-artifact.svg)](https://github.com/LLMServe/dLoRA-artifact.git) | - | OSDI'24 |
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
