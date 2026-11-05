"""Fixed-horizon per-domain cycles with a hard per-image repetition ceiling."""
import numpy as np


class CappedCycleSampler:
    def __init__(self, n, seed, probabilities=None, domains=None, batch_size=16,
                 source_quota=15, steps=6000, maximum_repeats=6):
        if (not isinstance(n, int) or n < 1 or not isinstance(steps, int) or steps < 1
                or not isinstance(maximum_repeats, int) or maximum_repeats < 1
                or not 0 < source_quota < batch_size):
            raise ValueError("Invalid fixed-horizon sampling configuration")
        self.n, self.seed, self.epoch, self._start = n, seed, 0, 0
        self.batch_size, self.source_quota = batch_size, source_quota
        self.domains = np.asarray(domains)
        if (self.domains.shape != (n,) or self.domains.dtype.kind not in "iu"
                or set(self.domains.tolist()) != {0, 1}):
            raise ValueError("Both source and phone domains must be represented")
        if probabilities is not None:
            weights = np.asarray(probabilities, dtype=float)
            if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
                raise ValueError("Invalid image sampling probabilities")
            for domain in (0, 1):
                values = weights[self.domains == domain]
                if not np.allclose(values, values[0]):
                    raise ValueError("Capped cycles require uniform sampling within each domain")
        streams = []
        for domain, quota in ((0, source_quota), (1, batch_size - source_quota)):
            indices = np.flatnonzero(self.domains == domain)
            draws = steps * quota
            if draws > len(indices) * maximum_repeats:
                raise ValueError(f"Insufficient domain{domain} capacity for the hard repetition cap")
            rng = np.random.default_rng(np.random.SeedSequence([seed, domain]))
            cycles = (draws + len(indices) - 1) // len(indices)
            stream = np.concatenate([rng.permutation(indices) for _ in range(cycles)])[:draws]
            streams.append(stream.reshape(steps, quota))
        schedule = np.concatenate(streams, axis=1)
        positions = np.random.default_rng(np.random.SeedSequence([seed, 2]))
        for row in schedule:
            positions.shuffle(row)
        self.schedule = schedule.ravel()
        if np.bincount(self.schedule, minlength=n).max() > maximum_repeats:
            raise ValueError("Generated schedule violates maximum repetition count")

    def __len__(self):
        return len(self.schedule)

    def __iter__(self):
        start, self._start = self._start, 0
        return iter(self.schedule[start:].tolist())

    def set_epoch(self, epoch):
        if epoch != 0:
            raise ValueError("Fixed-horizon capped schedule cannot restart at another epoch")

    def start_at(self, offset):
        if not isinstance(offset, int) or not 0 <= offset <= len(self) or offset % self.batch_size:
            raise ValueError("Resume offset must be a valid complete-batch boundary")
        self._start = offset
