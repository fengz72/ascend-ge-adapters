from setuptools import setup

setup(
    name="npu_prefix_infer_attention_score",
    version="1.0.0",
    description="PyTorch binding for PrefixInferAttentionScore Ascend custom operator "
                "(standalone, aclnn bridge + torchair converter)",
    packages=["npu_prefix_infer_attention_score"],
    package_data={"npu_prefix_infer_attention_score": ["_csrc/*.cpp", "_csrc/*.h"]},
    python_requires=">=3.8",
    install_requires=["torch", "torch-npu"],
)
