import torch


class OmniCandidateBank:
    """List of candidate negatives for the omni distillation loss (student vs OmniRetriever teacher).

    Every training step adds the teacher vectors of the GT segments of the batch to the list.
    The teacher vectors are fixed (precomputed by OmniRetriever), so an old candidate never goes stale.

    When the list is full, it is mined: every candidate gets a hardness score, its highest cosine
    similarity to the student vectors of the current batch. The weakest candidates (the ones no student
    confuses with its own teacher) are dropped and the hardest are kept, up to `size`.

    It is a plain object, not an nn.Module: it is not trained, not saved in the checkpoints, and DDP does not
    try to synchronise it (every GPU keeps its own list, of its own length).
    """

    def __init__(
        self,
        size,   # maximum number of candidates kept
        dim,    # size of a vector (3584)
    ):
        self.size = size
        self.vectors = torch.zeros(0, dim)                 # (Q, dim) L2-normalised teacher vectors
        self.steps = torch.zeros(0, dtype=torch.long)      # (Q,) step (pool row) of every candidate
        self.keys = torch.zeros(0, dtype=torch.long)       # (Q,) segment id of every candidate

    def __len__(self):
        return self.vectors.shape[0]

    @torch.no_grad()
    def add(
        self,
        vectors,    # (M, dim) teacher vectors of the GT segments of the batch, L2-normalised
        steps,      # (M,) step (pool row) of every segment
        keys,       # (M,) unique id of every segment (HTStepDataset seg_key)
        students,   # (N, dim) student vectors of the batch, L2-normalised: they score the hardness
    ):
        # the list lives on the device of the batch (it starts empty on the CPU)
        device = vectors.device
        self.vectors, self.steps, self.keys = self.vectors.to(device), self.steps.to(device), self.keys.to(device)
        vectors = vectors.detach().float()

        # 1. skip the segments that are already in the list (a video comes back every epoch)
        is_new = ~torch.isin(keys, self.keys)
        vectors, steps, keys = vectors[is_new], steps[is_new], keys[is_new]

        # 2. append them
        all_vectors = torch.cat([self.vectors, vectors], dim=0)
        all_steps = torch.cat([self.steps, steps], dim=0)
        all_keys = torch.cat([self.keys, keys], dim=0)

        # 3. too many: keep the hardest candidates, drop the weak ones
        if all_vectors.shape[0] > self.size:
            students = students.detach().float()
            similarity = students @ all_vectors.t()          # (N, Q + M)
            hardness = similarity.max(dim=0).values          # (Q + M,) closest student of every candidate
            keep = hardness.topk(self.size).indices
            all_vectors, all_steps, all_keys = all_vectors[keep], all_steps[keep], all_keys[keep]

        self.vectors, self.steps, self.keys = all_vectors, all_steps, all_keys
