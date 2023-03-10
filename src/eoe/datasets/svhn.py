from typing import Tuple, List, Union

import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Subset

from eoe.datasets.bases import TorchvisionDataset
from eoe.utils.logger import Logger
from eoe.utils.transformations import ConditionalCompose


class ADSVHN(TorchvisionDataset):
    def __init__(self, root: str, normal_classes: List[int], nominal_label: int,
                 train_transform: transforms.Compose, test_transform: transforms.Compose,
                 raw_shape: Tuple[int, int, int], logger: Logger = None,
                 limit_samples: Union[int, List[int]] = np.infty,
                 train_conditional_transform: ConditionalCompose = None,
                 test_conditional_transform: ConditionalCompose = None):
        """ AD dataset for SVHN. Implements :class:`eoe.datasets.bases.TorchvisionDataset`. """
        super().__init__(
            root, normal_classes, nominal_label, train_transform, test_transform, 10, raw_shape, logger, limit_samples,
            train_conditional_transform, test_conditional_transform
        )

        self._train_set = SVHN(
            self.root, split='train', download=True, transform=self.train_transform,
            target_transform=self.target_transform, conditional_transform=self.train_conditional_transform
        )
        self._train_set = self.create_subset(self._train_set, self._train_set.labels)
        self._test_set = SVHN(
            root=self.root, split='test', download=True, transform=self.test_transform,
            target_transform=self.target_transform, conditional_transform=self.test_conditional_transform
        )
        self._test_set = Subset(self._test_set, list(range(len(self._test_set))))

    def _get_raw_train_set(self):
        train_set = SVHN(
            self.root, train=True, download=True,
            transform=transforms.Compose([transforms.Resize((self.raw_shape[-1])), transforms.ToTensor(), ]),
            target_transform=self.target_transform
        )
        return Subset(
            train_set,
            np.argwhere(
                np.isin(np.asarray(train_set.labels), self.normal_classes)
            ).flatten().tolist()
        )


class SVHN(torchvision.datasets.SVHN):
    def __init__(self, *args, conditional_transform: ConditionalCompose = None, **kwargs):
        """
        Reimplements torchvision's SVHN s.t. it handles the optional conditional transforms.
        See :class:`eoe.datasets.bases.TorchvisionDataset`. Apart from this, the implementation doesn't differ from the
        standard one.
        """
        super(SVHN, self).__init__(*args, **kwargs)
        # print(*kwargs)
        # print("----------------------------")
        # super(SVHN,self).__init__(*args)
        self.conditional_transform = conditional_transform
        self.pre_transform, self.post_transform = None, None
        if self.transform is not None and self.conditional_transform is not None:
            totensor_pos = [isinstance(t, transforms.ToTensor) for t in self.transform.transforms]
            totensor_pos = totensor_pos.index(True) if True in totensor_pos else 0
            self.pre_transform = transforms.Compose(self.transform.transforms[:totensor_pos])
            self.post_transform = transforms.Compose(self.transform.transforms[totensor_pos:])

    def __getitem__(self, index) -> Tuple[torch.Tensor, int, int]:
        img, target = self.data[index], self.labels[index]
        if self.transform is None or isinstance(self.transform, transforms.Compose) and len(
                self.transform.transforms) == 0:
            img = img.float().div(255).unsqueeze(0)
        else:
            img = Image.fromarray(img.numpy(), mode="L")
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            if self.conditional_transform is not None:
                img = self.pre_transform(img)
                img = self.conditional_transform(img, target)
                img = self.post_transform(img)
            else:
                img = self.transform(img)
        return img, target, index
