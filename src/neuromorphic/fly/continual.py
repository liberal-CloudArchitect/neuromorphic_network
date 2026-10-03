"""Class-incremental odor learning with mushroom-body style learners (F1a).

All learners share one interface: ``fit_task`` sees only the current task's
samples, and ``predict`` restricts the decision to classes seen so far.  The
protocol is frozen in ``docs/fly/F1a-mb-protocol.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class OdorTask:
    """Synthetic glomerular odors expanded to per-PN activity.

    ``train_mode``/``test_mode`` identify the source prototype of each sample
    when a class is the union of several odors (multi-modal tasks).
    """

    train_x: Tensor
    train_y: Tensor
    test_x: Tensor
    test_y: Tensor
    n_classes: int
    train_mode: Tensor | None = None
    test_mode: Tensor | None = None


def _draw_prototypes(
    rng: np.random.Generator, count: int, n_glomeruli: int, active_probability: float
) -> FloatArray:
    active = rng.random((count, n_glomeruli)) < active_probability
    return np.where(active, rng.lognormal(0.0, 0.5, (count, n_glomeruli)), 0.0)


def _draw_samples(
    prototypes: FloatArray,
    source: IntArray,
    *,
    sigma: float,
    pn_glomerulus_index: IntArray,
    pn_noise: float,
    rng: np.random.Generator,
) -> Tensor:
    base = prototypes[source]
    glomeruli = base * (1.0 + sigma * rng.standard_normal(base.shape))
    glomeruli = np.maximum(glomeruli + 0.5 * sigma * rng.standard_normal(base.shape), 0.0)
    pn = glomeruli[:, pn_glomerulus_index]
    pn = np.maximum(pn + pn_noise * rng.standard_normal(pn.shape), 0.0)
    return torch.as_tensor(pn, dtype=torch.float32)


def make_odor_task(
    pn_glomerulus_index: IntArray,
    *,
    n_glomeruli: int,
    n_classes: int,
    n_train: int,
    n_test: int,
    sigma: float,
    rng: np.random.Generator,
    active_probability: float = 0.3,
    pn_noise: float = 0.05,
    families: int = 0,
    family_mixing: float = 0.0,
) -> OdorTask:
    """Draw class prototypes over glomeruli and noisy samples around them.

    With ``families > 0`` each class ``c`` mixes its own prototype with the
    shared prototype of family ``c % families``; ``family_mixing`` in [0, 1)
    controls how similar odors of one family are, and families span tasks so
    that later tasks contain look-alikes of earlier classes.
    """

    if n_classes <= 0 or n_train <= 0 or n_test <= 0 or sigma < 0:
        raise ValueError("invalid odor task size or noise")
    prototypes = _draw_prototypes(rng, n_classes, n_glomeruli, active_probability)
    if families:
        if not 0.0 <= family_mixing < 1.0:
            raise ValueError("family_mixing must be in [0, 1)")
        family = _draw_prototypes(rng, families, n_glomeruli, active_probability)
        shared = family[np.arange(n_classes) % families]
        prototypes = (1.0 - family_mixing) * prototypes + family_mixing * shared

    def draw(per_class: int) -> tuple[Tensor, Tensor]:
        labels = np.repeat(np.arange(n_classes), per_class)
        samples = _draw_samples(
            prototypes,
            labels,
            sigma=sigma,
            pn_glomerulus_index=pn_glomerulus_index,
            pn_noise=pn_noise,
            rng=rng,
        )
        return samples, torch.as_tensor(labels)

    train_x, train_y = draw(n_train)
    test_x, test_y = draw(n_test)
    return OdorTask(train_x, train_y, test_x, test_y, n_classes)


def make_multimodal_task(
    pn_glomerulus_index: IntArray,
    *,
    n_glomeruli: int,
    n_classes: int,
    modes: int,
    n_train_per_mode: int,
    n_test_per_mode: int,
    sigma: float,
    rng: np.random.Generator,
    active_probability: float = 0.3,
    pn_noise: float = 0.05,
) -> OdorTask:
    """Each class is the union of ``modes`` independently drawn odors.

    Prototype ``m`` belongs to class ``m // modes``, so the class mean in PN
    space blends unrelated odors while each odor stays a tight cluster.
    """

    if min(n_classes, modes, n_train_per_mode, n_test_per_mode) <= 0 or sigma < 0:
        raise ValueError("invalid multi-modal task size or noise")
    prototypes = _draw_prototypes(rng, n_classes * modes, n_glomeruli, active_probability)

    def draw(per_mode: int) -> tuple[Tensor, Tensor, Tensor]:
        source = np.repeat(np.arange(n_classes * modes), per_mode)
        samples = _draw_samples(
            prototypes,
            source,
            sigma=sigma,
            pn_glomerulus_index=pn_glomerulus_index,
            pn_noise=pn_noise,
            rng=rng,
        )
        return samples, torch.as_tensor(source // modes), torch.as_tensor(source)

    train_x, train_y, train_mode = draw(n_train_per_mode)
    test_x, test_y, test_mode = draw(n_test_per_mode)
    return OdorTask(train_x, train_y, test_x, test_y, n_classes, train_mode, test_mode)


def kc_code(pn: Tensor, weights: Tensor, *, sparsity: float) -> Tensor:
    """Binary KC code: the top ``sparsity`` fraction of positively driven KCs.

    ``weights`` is ``[kc, pn]``.  The APL-like k-winners-take-all keeps exactly
    ``k`` winners unless fewer KCs receive positive drive.
    """

    if not 0.0 < sparsity <= 1.0:
        raise ValueError("sparsity must be in (0, 1]")
    drive = pn @ weights.T
    k = max(1, round(sparsity * weights.shape[0]))
    top = torch.topk(drive, k, dim=1).indices
    code = torch.zeros_like(drive)
    code.scatter_(1, top, 1.0)
    return code * (drive > 0)


class Learner(Protocol):
    def fit_task(self, x: Tensor, y: Tensor) -> None: ...

    def predict(self, x: Tensor, seen: int) -> Tensor: ...


class AssociativeReadout:
    """Strengthen only active-KC -> correct-class synapses; all others stay frozen."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        self.weight = torch.zeros(n_classes, n_features)

    def fit_task(self, x: Tensor, y: Tensor) -> None:
        self.weight.index_add_(0, y, x)

    def predict(self, x: Tensor, seen: int) -> Tensor:
        return (x @ self.weight[:seen].T).argmax(dim=1)


class PerceptronReadout:
    """Multiclass perceptron: errors also depress the wrongly winning class."""

    def __init__(self, n_features: int, n_classes: int, *, epochs: int, seed: int) -> None:
        self.weight = torch.zeros(n_classes, n_features)
        self.epochs = epochs
        self.generator = torch.Generator().manual_seed(seed)
        self.seen = 0

    def fit_task(self, x: Tensor, y: Tensor) -> None:
        self.seen = max(self.seen, int(y.max()) + 1)
        for _ in range(self.epochs):
            for i in torch.randperm(x.shape[0], generator=self.generator).tolist():
                predicted = int((self.weight[: self.seen] @ x[i]).argmax())
                target = int(y[i])
                if predicted != target:
                    self.weight[target] += x[i]
                    self.weight[predicted] -= x[i]

    def predict(self, x: Tensor, seen: int) -> Tensor:
        return (x @ self.weight[:seen].T).argmax(dim=1)


class NearestClassMean:
    """Cosine nearest-class-mean over raw features (prototype learning)."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        self.total = torch.zeros(n_classes, n_features)
        self.count = torch.zeros(n_classes)

    def fit_task(self, x: Tensor, y: Tensor) -> None:
        self.total.index_add_(0, y, x)
        self.count.index_add_(0, y, torch.ones_like(y, dtype=torch.float32))

    def predict(self, x: Tensor, seen: int) -> Tensor:
        means = self.total[:seen] / self.count[:seen].clamp(min=1).unsqueeze(1)
        means = nn.functional.normalize(means, dim=1)
        return (nn.functional.normalize(x, dim=1) @ means.T).argmax(dim=1)


class NearestNeighbor:
    """Cosine 1-nearest-neighbour over every stored training sample.

    A non-parametric reference: its memory grows with the number of samples,
    unlike the fixed-size associative readout.
    """

    def __init__(self) -> None:
        self.x: list[Tensor] = []
        self.y: list[Tensor] = []

    def fit_task(self, x: Tensor, y: Tensor) -> None:
        self.x.append(nn.functional.normalize(x, dim=1))
        self.y.append(y)

    def predict(self, x: Tensor, seen: int) -> Tensor:
        stored = torch.cat(self.x)
        labels = torch.cat(self.y)
        keep = labels < seen
        similarity = nn.functional.normalize(x, dim=1) @ stored[keep].T
        return labels[keep][similarity.argmax(dim=1)]


class GradientLearner:
    """Fine-tune a softmax model on each task in turn (no replay, no regularizer)."""

    def __init__(
        self,
        model: nn.Module,
        *,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 1e-3,
        seed: int,
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.generator = torch.Generator().manual_seed(seed)

    def fit_task(self, x: Tensor, y: Tensor) -> None:
        self.model.train()
        for _ in range(self.epochs):
            order = torch.randperm(x.shape[0], generator=self.generator)
            for start in range(0, x.shape[0], self.batch_size):
                batch = order[start : start + self.batch_size]
                loss = nn.functional.cross_entropy(self.model(x[batch]), y[batch])
                self.optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                self.optimizer.step()

    @torch.no_grad()
    def predict(self, x: Tensor, seen: int) -> Tensor:
        self.model.eval()
        logits: Tensor = self.model(x)
        return logits[:, :seen].argmax(dim=1)


@dataclass(frozen=True, slots=True)
class IncrementalResult:
    """``accuracy[t, j]``: accuracy on task j after training through task t."""

    accuracy: FloatArray

    @property
    def final_accuracy(self) -> float:
        return float(self.accuracy[-1].mean())

    @property
    def forgetting(self) -> float:
        tasks = self.accuracy.shape[0]
        if tasks < 2:
            return 0.0
        learned = np.diag(self.accuracy)[:-1]
        return float((learned - self.accuracy[-1, :-1]).mean())


def run_class_incremental(
    learner: Learner,
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
    test_y: Tensor,
    *,
    classes_per_task: int,
) -> IncrementalResult:
    """Train task by task and record per-task test accuracy after each task."""

    n_classes = int(train_y.max()) + 1
    if n_classes % classes_per_task:
        raise ValueError("classes must split evenly into tasks")
    tasks = n_classes // classes_per_task
    task_of_train = train_y // classes_per_task
    task_of_test = test_y // classes_per_task
    accuracy = np.zeros((tasks, tasks))
    for t in range(tasks):
        mask = task_of_train == t
        learner.fit_task(train_x[mask], train_y[mask])
        seen = (t + 1) * classes_per_task
        for j in range(t + 1):
            selected = task_of_test == j
            predicted = learner.predict(test_x[selected], seen)
            accuracy[t, j] = float((predicted == test_y[selected]).float().mean())
    return IncrementalResult(accuracy)
