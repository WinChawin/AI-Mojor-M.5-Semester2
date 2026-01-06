import torch

x = torch.rand(3, 4, dtype=torch.float32)
# print(x)

# tensor([[0.1644, 0.7608, 0.3564, 0.9114],
#         [0.2579, 0.5691, 0.4266, 0.1673],
#         [0.8408, 0.2796, 0.3800, 0.0977]])

x = torch.zeros(5, 3, dtype=torch.long)
# print(x)
# tensor([[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]])

x = torch.zeros(5, 3, dtype=torch.long)
x = x.new_ones(5, 3, dtype=torch.float64)
# print(x)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64)

x = torch.tensor([4.123, 3])
# print(x)
# tensor([4.1230, 3.0000])

x = torch.zeros(5, 3, dtype=torch.long)
x = torch.randn_like(x, dtype=torch.float32)
# print(x)
# tensor([[ 1.2431, -1.6546, -1.1400],
#         [ 0.2930, -0.2959, -1.7899],
#         [-1.2062, -0.3146, -0.9580],
#         [ 0.7348,  0.3988,  0.9379],
#         [ 0.1994,  1.4759,  0.4457]])

x = torch.Tensor(5, 3).fill_(7)
# print(x)
# tensor([[7., 7., 7.],
#         [7., 7., 7.],
#         [7., 7., 7.],
#         [7., 7., 7.],
#         [7., 7., 7.]])

x = torch.Tensor(5, 3).fill_(7).type(torch.int32)
# print(x)
# tensor([[7, 7, 7],
#         [7, 7, 7],
#         [7, 7, 7],
#         [7, 7, 7],
#         [7, 7, 7]], dtype=torch.int32)

torch.manual_seed(1)
x = torch.rand(5, 3)
# print(x)
# tensor([[0.7576, 0.2793, 0.4031],
#         [0.7347, 0.0293, 0.7999],
#         [0.3971, 0.7544, 0.5695],
#         [0.4388, 0.6387, 0.5247],
#         [0.6826, 0.3051, 0.4635]]) เสมอ จำแล้ว

