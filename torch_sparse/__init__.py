import torch


def coalesce(indices, values, m=None, n=None):
    if indices.numel() == 0:
        return indices, values

    size = (
        int(m) if m is not None else int(indices[0].max().item()) + 1,
        int(n) if n is not None else int(indices[1].max().item()) + 1,
    )
    sparse = torch.sparse_coo_tensor(indices, values, size=size).coalesce()
    return sparse.indices(), sparse.values()


def spmm(indices, values, m, n, matrix):
    sparse = torch.sparse_coo_tensor(indices, values, size=(int(m), int(n))).coalesce()
    return torch.sparse.mm(sparse, matrix)
