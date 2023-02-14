# Selective Deeply Supervised Multi-Scale Attention Network for Brain Tumor Segmentation


Brain tumors are one of the deadliest forms of cancer, characterized by the abnormal 1
proliferation of brain cells. While early identification of brain tumors can greatly aid in their therapy, 2
the process of manual segmentation performed by expert doctors, which is often time-consuming, 3
tedious, and prone to human error, can act as a bottleneck in the diagnostic process. This motivates 4
the development of automated algorithms for brain tumor segmentation. This process is however 5
complicated due to the difficulty in accurately segmenting the enhanced and core tumor regions due 6
to the high levels of inter- and intra-tumor heterogeneity in terms of texture, morphological structure, 7
and shape. This study proposes a fully automatic method called Selective Deeply Supervised Multi- 8
Scale Attention Network (SDS-MSA-Net) for segmenting brain tumor regions using a multi-scale 9
attention network with novel Selective Deep Supervision (SDS) mechanisms for training. The method 10
utilizes a 3D input composed of five consecutive slices, in addition to a 2D slice, to maintain sequential 11
information. The proposed multi-scale architecture includes two encoding units to extract meaningful 12
global and local features from the 3D and 2D inputs, respectively. These coarse features are then 13
passed through attention units to filter out redundant information by assigning lower weights. The 14
refined features are fed into a decoder block, which upscales the features at various levels while 15
learning patterns relevant to all tumor regions.The SDS block is introduced to immediately upscale 16
features from intermediate layers of the decoder, with the aim of producing segmentations of the 17
whole, enhanced, and core tumor regions. The proposed framework was evaluated on the BraTS2020 18
dataset and showed improved performance in brain tumor region segmentation, particularly in 19
the segmentation of the core and enhancing tumor regions, demonstrating the effectiveness of the 20
proposed approach

