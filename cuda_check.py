import torch

def check_cuda():
    """
    检查 CUDA 是否可用，并打印相关信息。
    """
    if torch.cuda.is_available():
        print("CUDA is available! 🎉")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        # 尝试创建一个张量并将其移动到 GPU，以进一步确认功能正常
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"Successfully moved a tensor to GPU: {test_tensor}")
            print("Your CUDA setup appears to be working correctly!")
        except Exception as e:
            print(f"Error testing CUDA device: {e}")
            print("CUDA is available, but there might be an issue with device access or functionality.")
    else:
        print("CUDA is not available. 😔")
        print("Please check your GPU drivers and PyTorch installation.")
        print("You might be running on CPU.")

if __name__ == "__main__":
    check_cuda()
