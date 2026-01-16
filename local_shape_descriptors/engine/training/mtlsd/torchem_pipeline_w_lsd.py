import numpy as np
import torch.nn as nn
import torch_em
import torch_em.data.datasets as torchem_data
from torch_em.model import AnisotropicUNet
from torch_em.util.debug import check_loader, check_trainer
from hemi import get_hemi_loader, get_hemi_dataset
# from lsd import local_shape_descriptors


def get_loss(loss_name):
    loss_names = ["bce", "ce", "dice"]
    if isinstance(loss_name, str):
        assert loss_name in loss_names, f"{loss_name}, {loss_names}"
        if loss_name == "dice":
            loss_function = torch_em.loss.DiceLoss()
        elif loss == "ce":
            loss_function = nn.CrossEntropyLoss()
        elif loss == "bce":
            loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = loss_name

    # we need to add a loss wrapper for affinities
    if affinities:
        loss_function = torch_em.loss.LossWrapper(
            loss_function, transform=torch_em.loss.ApplyAndRemoveMask()
        )
    return loss_function


def main():
    # to input custom datasets to the network, must be set to None
    preconfigured_dataset = None

    # TODO make yacs based for now copied from notebook
    # train_data_paths = train_label_paths = val_data_paths = val_label_paths = "./training_data/snemi/snemi_train.h5"
    # data_key = "volumes/raw"
    # label_key = "volumes/labels/neuron_ids"
    patch_shape = (196, 196, 196)  # hemi brain

    # Whether to add a foreground channel (1 for all labels that are not zero) to the target.
    foreground = False
    # Whether to add affinity channels (= directed boundaries) or a boundary channel to the target.
    # Note that you can choose at most of these two options.
    affinities = True
    boundaries = False

    # the pixel offsets that are used to compute the affinity channels - this is neighborhood in lsd code
    offsets = [
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
        [-2, 0, 0], [0, -3, 0], [0, 0, -3],
        [-3, 0, 0], [0, -9, 0], [0, 0, -9]
    ]

    # arguments
    assert not (affinities and boundaries), "Predicting both affinities and boundaries is not supported"

    label_transform, label_transform2 = None, None
    if affinities:
        label_transform2 = torch_em.transform.label.AffinityTransform(
            offsets=offsets, add_binary_target=foreground, add_mask=True
        )
    elif boundaries:
        label_transform = torch_em.transform.label.BoundaryTransform(
            add_binary_target=foreground
        )
    elif foreground:
        label_transform = torch_em.transform.label.labels_to_binary

    batch_size = 1
    loss = "dice"
    metric = "dice"

    loss_function = get_loss(loss)
    metric_function = get_loss(metric)

    kwargs = dict(
        ndim=3, patch_shape=patch_shape, batch_size=batch_size,
        label_transform=label_transform, label_transform2=label_transform2
    )

    train_samples = ("EB-INNER", "EB-OUTER", "FB-INNER", "FB-OUTER", "LH", "LOBULA", "PB1", "PB2")
    val_samples = ("FB-INNER", "FB-OUTER", "LH", "LOBULA", "PB1", "PB2",)

    # now this is awful, if you have mention a np array for every matrix. but we do it as this now as per example
    train_rois = (
        np.s_[:, :, :], np.s_[:, :, :], np.s_[:-75, :, :], np.s_[:-75, :, :], np.s_[:-75, :, :], np.s_[:-75, :, :],
        np.s_[:-75, :, :], np.s_[:-75, :, :])
    val_rois = (
        np.s_[-75:, :, :], np.s_[-75:, :, :], np.s_[-75:, :, :], np.s_[-75:, :, :], np.s_[-75:, :, :],
        np.s_[-75:, :, :],)
    train_loader = torchem_data.get_hemi_loader(samples=train_samples, rois=train_rois, **kwargs)
    val_loader = torchem_data.get_hemi_loader(samples=val_samples, rois=val_rois, **kwargs)

    # example for isotropic scaling with a depth of 4
    scale_factors = 4 * [[2, 2, 2]]

    # example for 4 levels with anisotropic scaling in the first two levels (scale only in xy)
    # scale_factors = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]

    initial_features = 32
    final_activation = None

    # If you leave the in/out_channels as None an attempt will be made to automatically deduce these numbers.
    in_channels = 1
    out_channels = None

    if final_activation is None and loss == "dice":
        final_activation = "Sigmoid"
    print("Adding a sigmoid activation because we are using dice loss")

    if in_channels is None:
        in_channels = 1

    if out_channels is None:
        if affinities:
            n_off = len(offsets)
            out_channels = n_off + 1 if foreground else n_off
        elif boundaries:
            out_channels = 2 if foreground else 1
        elif foreground:
            out_channels = 1
        assert out_channels is not None, "The number of out channels could not be deduced automatically. Please set it manually in the cell above."

    print("Creating 3d UNet with", in_channels, "input channels and", out_channels, "output channels.")
    model = AnisotropicUNet(
        in_channels=in_channels, out_channels=out_channels, scale_factors=scale_factors,
        final_activation=final_activation
    )

    experiment_name = "hemi-torchem"
    n_iterations = 10000
    learning_rate = 1.0e-4

    trainer = torch_em.default_segmentation_trainer(
        name=experiment_name, model=model,
        train_loader=train_loader, val_loader=val_loader,
        loss=loss_function, metric=metric_function,
        learning_rate=learning_rate,
        mixed_precision=True,
        log_image_interval=50,
        # logger=None
    )
    trainer.fit(n_iterations)


if __name__ == '__main__':
    main()
