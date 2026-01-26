import torch

print("=== PyTorch バックエンド確認 ===")
print(f"PyTorch version: {torch.__version__}")

# CUDA (NVIDIA GPU) の確認
if torch.cuda.is_available():
    print(f"CUDA: 利用可能")
    print(f"  - CUDA version: {torch.version.cuda}")
    print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
# MPS (Apple Silicon Mac) の確認
elif torch.backends.mps.is_available():
    print("MPS: 利用可能")
    device = torch.device("mps")
# CPU のみ
else:
    print("GPU: 利用不可（CPUを使用します）")
    if not torch.backends.mps.is_built():
        print("  - MPS: PyTorchがMPS無効でビルドされています（Windowsでは正常）")
    device = torch.device("cpu")

print(f"\n使用デバイス: {device}")

# 動作確認
x = torch.ones(5, device=device)
y = x * 2
print(f"動作確認OK: {y}")