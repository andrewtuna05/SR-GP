# Single-Image Super-Resolution: Exploratory Modifications

This jupyter notebook implements and extends the Single Image Super Resolution (SISR) algorithm demonstrated in this [paper](https://hhexiy.github.io/docs/papers/srgpr.pdf) by He He and Wan-Chi Siu.

The motivation for this passion project was to self-explore and experiment with Gaussian Process–based reconstruction methods to develop a simple framework for image reconstruction.
I wanted to experiment with what I learned during my NSF REU at Cal Poly Pomona to see if any minor improvements could be made. Any changes and adaptations were purely exploratory!

This notebook provides a performance comparison between color images reconstructed with standard Bicubic Interpolation versus the Generalized Cauchy kernel through metrics such as PSNR and SSIM.

In the future, I plan to revisit this and incorporate additional features including: differnet exotic kernels, joint RGB modeling, and Linear Coregionalization Models.

Credit to Dr. Jimmy Risk for [easyGPR_helper.py](https://github.com/jimmyrisk/EasyGPR)