# Hardware Evaluation

## Setup

### [NVIDIA Jetson Orin Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

NVIDIA Jetson Orin Nano was configured with newest [JetPack 5.1.1](https://developer.nvidia.com/embedded/jetpack), following [Getting Started Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit) and [User Guide](https://developer.nvidia.com/embedded/learn/jetson-orin-nano-devkit-user-guide/index.html).

Device configuration:
- hostname: `put-orin-nano`
- username: `put`
- password: `put`

**ONNX Runtime** was built in the newest version (1.16.0) in accordance with instructions provided in [_Build ONNX Runtime with Execution Providers_](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) for _NVIDIA Jetson TX1/TX2/Nano/Xavier_ with CPU, CUDA and TensorRT providers.

<p align="center">
  <img width=400 src="https://developer.nvidia.com/sites/default/files/akamai/embedded/images/jetson_orin_nano/NVIDIA-JetsonOrinNano_devkit-Fornt-Angle-top.jpg" />
</p>


### [Intel NUC 12WSHi3](https://www.intel.com/content/www/us/en/support/articles/000005522/intel-nuc.html)

The hardware setup was done according to the instructions:
- [User Guide](https://www.intel.com/NUC12WS-Support)
- [Integration Guide](https://www.intel.com/content/dam/support/us/en/documents/intel-nuc/NUC12WSK_NUC12WSH_IntegrationGuide.pdf)

[Ubuntu 22.04.2 LTS](https://releases.ubuntu.com/jammy/) Desktop image was used as an operating system with the below device configuration:
- hostname: `put-nuc`
- username: `put`
- password: `put`

**ONNX Runtime** was built with CPU and VPU support in the below section.

<p align="center">
  <img width=400 src="https://a.allegroimg.com/original/11b798/3ef2a4f8486abe0a9b5eba7ec69f/Intel-NUC12WSHI3-i3-1220P-8GB-120GB-SSD-W10-11P" />
</p>


### [Intel MYRIAD VPU (VEGA-320)](https://www.advantech.com/en/products/3d060f1e-e73e-460d-b38c-c69f76312c91/vega-320/mod_f8aaa5f2-fe32-4a58-b5b4-2a02a857852a)

Intel MYRIAD VPU as a co-processor extends computational capabilities of Intel NUC, so it was configured with Intel NUC 12WSHi3, and [**ONNX Runtime**](https://onnxruntime.ai/docs/build/eps.html#openvino) backed by [**OpenVINO**](https://docs.openvino.ai/2023.0/openvino_docs_install_guides_overview.html?ENVIRONMENT=DEV_TOOLS&OP_SYSTEM=WINDOWS&VERSION=v_2023_0_1&DISTRIBUTION=PIP) with MYRIAD VPU support.

<p align="center">
  <img width=200 src="https://advdownload.advantech.com/productfile/PIS/VEGA-320/Product%20-%20Photo(B)/VEGA-320_3D_S20200327134445.jpg" />
</p>

