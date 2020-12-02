import torch

import torch.nn.functional as F

aaa = torch.Tensor(
    [311, 26429, 17214, 351, 2723, 576, 1560, 289, 228, 737, 2794, 983, 845, 2012, 1561, 6768, 5306, 3116, 405, 4127,
     2978, 1396, 45])

bbb = aaa/torch.sum(aaa)
print(bbb)