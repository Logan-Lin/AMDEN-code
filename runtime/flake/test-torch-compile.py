import torch 

@torch.compile
def test_function(x):
    return torch.mean(x)


x = torch.rand((100, 100)).to("cuda")
print(test_function(x))
