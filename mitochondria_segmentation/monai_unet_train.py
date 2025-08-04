"""
Trains standard UNet from Monai.
Co-developed with Gemini 2.5 Pro.
Author: Samia Mohinta
Affiliation: University of Cambridge, UK


"""

# --- Main Training Function ---
import argparse
import numpy as np
import cv2
import os
from datetime import datetime

import torch.nn as nn
import torch
import torchmetrics
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, WeightedRandomSampler

# Additional imports for EMDataset
import glob
import zarr

try:
    import scipy.ndimage
except ImportError:
    print("Warning: scipy.ndimage not found. 3D image processing might fall back to 2D slice-by-slice operations.")

# --- NEW: Import MONAI UNet and ResidualUnit ---
from monai.networks.nets import UNet
from monai.networks.blocks import ResidualUnit  # If you want to use ResidualUnit for custom blocks

from torch.utils.tensorboard import SummaryWriter
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Insert at the beginning for higher priority

# Now, you can import modules from the 'engine' package
from engine.dataset.dataset import EMDataset
from engine.models.rsunet import RSUNet
from engine.utils.utils import print_model_parameters

# If losses are in engine/utils/losses.py:
from engine.utils.losses import DiceLoss, DiceCE


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Make train ckpt directory
    exp_name = args.exp_name
    ckpt_dir = "./train_logs_ckpt"
    exp_name = os.path.join(ckpt_dir, exp_name)
    if not os.path.isdir(str(exp_name)):
        os.makedirs(str(exp_name))

    # --- Initialize TensorBoard SummaryWriter ---
    # The log_dir will be a subdirectory within your exp_name folder
    log_dir = os.path.join(exp_name, "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    print("\n--- Setting up Datasets and DataLoaders ---")

    train_dataset = EMDataset(
        zarr_data_dirs=args.train_zarr_dirs,
        label_type=args.label_type,  # Targeting mitochondria
        patch_size=args.patch_size,
        stride=args.stride,
        original_resolution_nm=args.original_res,
        target_resolution_nm=args.target_res,
        clahe=args.clahe,
        subsample_frac=args.subsample_frac,
        subsample_number=args.subsample_number,
        seed=args.subsample_seed,
        balance_patches=args.balance_patches,
        min_positive_pixels=args.min_positive_pixels
    )

    if train_dataset.balance_patches and train_dataset.patch_weights is not None:
        sampler = WeightedRandomSampler(
            weights=train_dataset.patch_weights,
            num_samples=len(train_dataset.patch_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True)
        print("Train DataLoader created with WeightedRandomSampler.")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
        print("Train DataLoader created without WeightedRandomSampler (balancing not enabled or weights not computed).")

    test_loader = None
    if args.test_zarr_dirs:
        test_dataset = EMDataset(
            zarr_data_dirs=args.test_zarr_dirs,
            label_type=args.label_type,
            patch_size=args.patch_size,
            stride=args.stride,
            original_resolution_nm=args.original_res,
            target_resolution_nm=args.target_res,
            clahe=args.clahe,
            subsample_frac=1.0,
            subsample_number=0
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)
        print("Test DataLoader created.")
    else:
        print("No test dataset specified.")

    print("\n--- Initializing Model, Loss, and Optimizer ---")

    # --- CHANGE THIS SECTION FOR MONAI UNet ---
    # Determine the number of spatial dimensions (3 for 3D)
    spatial_dims = 3
    # Input channels: 1 (grayscale EM images)
    in_channels = 1
    # Output channels: 1 (binary segmentation: foreground/background)
    out_channels = 1

    # MONAI UNet parameters:
    # `channels`: controls number of features maps at each level.
    # `strides`: controls spatial downsampling at each level (usually (2,2,2) for 3D).
    # `num_res_units`: number of residual units at each level if `adn_ordering` is 'adn'
    #                   or if `block_res` is True.
    # `norm`: normalization layer ('BATCH', 'INSTANCE', 'GROUP', None). 'BATCH' is common.
    # `adn_ordering`: activation, dropout, normalization order. Usually 'adn' or 'nad'.

    # Example MONAI UNet (similar structure to U-Net you have):
    # This roughly corresponds to your RSUNet's pooling levels.
    # RSUNet had 4 pooling levels, so 5 levels in total (input level + 4 downsamples).
    # Its channels were 28 -> 36 -> 48 -> 64 -> 80.
    # For MONAI, you typically specify `channels` and `strides` that define the levels.
    # `channels=(C_in, C_down1, C_down2, C_down3, C_bottleneck, C_up1, C_up2, C_up3, C_up4)`
    # The `strides` should match the pooling. E.g., for 4 pooling layers, 4 strides.

    # Let's try to match RSUNet's structure approximately:
    # Levels: L0 (input) -> L1 -> L2 -> L3 -> L4 (bottleneck)
    # Channels: 28 -> 36 -> 48 -> 64 -> 80 (bottleneck)
    # Strides: (2,2,2) for each downsampling.
    # So, `strides` should have length 4.
    # `channels` should have length 5 (number of levels).

    net = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(28, 36, 48, 64, 80),  # Matching your RSUNet's channel progression
        strides=(2, 2, 2, 2),  # 4 downsampling steps, each by factor of 2
        kernel_size=3,  # Default kernel size for conv layers in UNet (3x3x3 for 3D)
        up_kernel_size=3,  # Default kernel size for transpose conv (2x2x2 for 3D)
        num_res_units=0,  # Set to 0 for standard UNet (no residual units beyond default blocks)
        # Set to 1 or 2 for a ResUNet type behavior (more residual blocks)
        norm='batch',  # Use BatchNorm
        dropout=0.1  # No dropout
    )
    # If you want it to be more explicitly "ResUNet-like" in its blocks (beyond the Conv/Norm/Activations)
    # you might need to adjust `num_res_units` or explore other MONAI models like `DynUNet` or define custom blocks.
    # For now, this UNet with `norm='batch'` and `kernel_size=3` is a good isotropic baseline.

    model = net.to(device)

    # Print parameter count
    print_model_parameters(model, "Monai_UNet")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    criteria = DiceCE()

    acc_metric = torchmetrics.Accuracy(task='binary').to(device)
    ppv_metric = torchmetrics.Precision(task='binary').to(device)
    tpr_metric = torchmetrics.Recall(task='binary').to(device)
    iou_metric = torchmetrics.JaccardIndex(task='binary').to(device)

    acc_metric_test = torchmetrics.Accuracy(task='binary').to(device)
    ppv_metric_test = torchmetrics.Precision(task='binary').to(device)
    tpr_metric_test = torchmetrics.Recall(task='binary').to(device)
    iou_metric_test = torchmetrics.JaccardIndex(task='binary').to(device)

    run_losses, test_losses = [], []
    run_TPRs, test_TPRs = [], []
    run_PPVs, test_PPVs = [], []
    run_accs, test_accs = [], []
    run_IOUs, test_IOUs = [], []

    start_epoch = 0  # Default starting epoch
    best_iou = 0.0

    if args.resume_checkpoint:
        if not os.path.exists(args.resume_checkpoint):
            print(f"Warning: Resume checkpoint '{args.resume_checkpoint}' not found. Starting from scratch.")
        else:
            print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)

            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Restore epoch and best_iou
            start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
            best_iou = checkpoint.get('best_iou', 0.0)  # Use .get() for backward compatibility

            print(f"Resumed from Epoch {start_epoch}, with best IoU: {best_iou:.4f}")
            # If you were tracking run_losses, test_losses etc. in the checkpoint, load them too
            # e.g., run_losses = checkpoint.get('run_losses', [])
            # test_losses = checkpoint.get('test_losses', [])
            # You'd need to save these in the checkpoint dict as well.
            run_losses = checkpoint.get('run_losses', [])
            test_losses = checkpoint.get('test_losses', [])

    print("\n========== BEGIN TRAINING LOOP ==========")
    start_time = datetime.now()

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        model.train()
        run_loss = 0.0

        if (epoch == 50 and args.hotstart):
            for g in optimizer.param_groups:
                g['lr'] = args.learning_rate_after_hotstart_50
            print(f"Learning rate changed to {args.learning_rate_after_hotstart_50} at epoch 50.")

        for g in optimizer.param_groups:
            print(f"Current learning rate: {g['lr']}")

        for batch_idx, (sample, labels) in enumerate(train_loader):
            sample = sample.to(device)
            labels = labels.to(device)

            if args.rotation_augs:
                for z0 in range(sample.size(0)):
                    rot_n = np.random.randint(low=0, high=4)
                    sample[z0, :, :, :, :] = torch.rot90(sample[z0, :, :, :, :], rot_n, dims=[2, 3])
                    labels[z0, :, :, :, :] = torch.rot90(labels[z0, :, :, :, :], rot_n, dims=[2, 3])

            if args.contrast_augs:
                for z0 in range(sample.size(0)):
                    rand_int1 = np.random.randint(low=0, high=4)
                    if rand_int1 != 0:
                        alpha = torch.rand(1).to(device) * 1.0 + 0.5
                        beta = torch.rand(1).to(device) * 1.0 - 0.5
                        sample[z0, :, :, :, :] = torch.clamp(sample[z0, :, :, :, :] * alpha + beta, -1, 1)

            optimizer.zero_grad()
            pred_logits = model(sample)

            loss = criteria(pred_logits, labels.float(), loss_weights=args.loss_weights)

            loss.backward()
            optimizer.step()

            run_loss += loss.item()

            pred_probs = torch.sigmoid(pred_logits)
            pred_binary = (pred_probs >= 0.5).int()

            acc_metric.update(pred_binary, labels.int())
            ppv_metric.update(pred_binary, labels.int())
            tpr_metric.update(pred_binary, labels.int())
            iou_metric.update(pred_binary, labels.int())

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}, "
                      f"Acc: {acc_metric.compute().item():.4f}, PPV: {ppv_metric.compute().item():.4f}, "
                      f"TPR: {tpr_metric.compute().item():.4f}, IoU: {iou_metric.compute().item():.4f}")

            if args.save_images and epoch in [0, 10, 20, 50, 100, 150] and (batch_idx + 1) % (
                    len(train_loader) // 3 + 1) == 0:
                save_dir = os.path.join(exp_name, f"epoch_{epoch}", f"batch_{batch_idx}")
                os.makedirs(save_dir, exist_ok=True)

                pred_binary_np = pred_binary[0, 0].cpu().numpy() * 255
                labels_np = labels[0, 0].cpu().numpy() * 255
                sample_np = ((sample[0, 0].cpu().numpy() + 1) * 127.5).astype(np.uint8)

                for z_slice in [0, pred_binary_np.shape[0] // 2, pred_binary_np.shape[0] - 1]:
                    cv2.imwrite(os.path.join(save_dir, f"prediction_z{z_slice:03d}.png"), pred_binary_np[z_slice, :, :])
                    cv2.imwrite(os.path.join(save_dir, f"gt_z{z_slice:03d}.png"), labels_np[z_slice, :, :])
                    cv2.imwrite(os.path.join(save_dir, f"input_z{z_slice:03d}.png"), sample_np[z_slice, :, :])

        epoch_run_loss = run_loss / len(train_loader)
        epoch_run_acc = acc_metric.compute().item()
        epoch_run_tpr = tpr_metric.compute().item()
        epoch_run_ppv = ppv_metric.compute().item()
        epoch_run_iou = iou_metric.compute().item()

        run_losses.append(epoch_run_loss)
        run_accs.append(epoch_run_acc)
        run_TPRs.append(epoch_run_tpr)
        run_PPVs.append(epoch_run_ppv)
        run_IOUs.append(epoch_run_iou)

        # Log epoch-level training metrics to TensorBoard
        writer.add_scalar('Loss/train_epoch', epoch_run_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', epoch_run_acc, epoch)
        writer.add_scalar('Precision/train_epoch', epoch_run_ppv, epoch)
        writer.add_scalar('Recall/train_epoch', epoch_run_tpr, epoch)
        writer.add_scalar('IoU/train_epoch', epoch_run_iou, epoch)

        print(f"\nEpoch {epoch + 1} Training Summary:")
        print(f"  Avg Loss: {epoch_run_loss:.4f}, Avg Acc: {epoch_run_acc:.4f}, "
              f"Avg PPV: {epoch_run_ppv:.4f}, Avg TPR: {epoch_run_tpr:.4f}, Avg IoU: {epoch_run_iou:.4f}")

        acc_metric.reset()
        ppv_metric.reset()
        tpr_metric.reset()
        iou_metric.reset()

        if (epoch + 1) % args.eval_interval == 0 and test_loader:
            model.eval()
            test_loss = 0.0
            print("\n--- Starting Test Phase ---")
            with torch.no_grad():
                for batch_idx_test, (sample_test, labels_test) in enumerate(test_loader):
                    sample_test = sample_test.to(device)
                    labels_test = labels_test.to(device)

                    pred_logits_test = model(sample_test)
                    loss_test = criteria(pred_logits_test, labels_test.float(), loss_weights=args.loss_weights)
                    test_loss += loss_test.item()

                    pred_probs_test = torch.sigmoid(pred_logits_test)
                    pred_binary_test = (pred_probs_test >= 0.5).int()

                    acc_metric_test.update(pred_binary_test, labels_test.int())
                    ppv_metric_test.update(pred_binary_test, labels_test.int())
                    tpr_metric_test.update(pred_binary_test, labels_test.int())
                    iou_metric_test.update(pred_binary_test, labels_test.int())

                epoch_test_loss = test_loss / len(test_loader)
                epoch_test_acc = acc_metric_test.compute().item()
                epoch_test_tpr = tpr_metric_test.compute().item()
                epoch_test_ppv = ppv_metric_test.compute().item()
                epoch_test_iou = iou_metric_test.compute().item()

                test_losses.append(epoch_test_loss)
                test_accs.append(epoch_test_acc)
                test_TPRs.append(epoch_test_tpr)
                test_PPVs.append(epoch_test_ppv)
                test_IOUs.append(epoch_test_iou)

                # Log epoch-level test metrics to TensorBoard
                writer.add_scalar('Loss/test_epoch', epoch_test_loss, epoch)
                writer.add_scalar('Accuracy/test_epoch', epoch_test_acc, epoch)
                writer.add_scalar('Precision/test_epoch', epoch_test_ppv, epoch)
                writer.add_scalar('Recall/test_epoch', epoch_test_tpr, epoch)
                writer.add_scalar('IoU/test_epoch', epoch_test_iou, epoch)

                print(f"Epoch {epoch + 1} Test Summary:")
                print(f"  Avg Loss: {epoch_test_loss:.4f}, Avg Acc: {epoch_test_acc:.4f}, "
                      f"Avg PPV: {epoch_test_ppv:.4f}, Avg TPR: {epoch_test_tpr:.4f}, Avg IoU: {epoch_test_iou:.4f}")

                if epoch_test_iou > best_iou:
                    #                     best_iou = epoch_test_iou
                    #                     model_save_path = os.path.join(exp_name, f'trained_model_epoch{epoch+1}_IoU{best_iou:.4f}.pth')
                    #                     torch.save(model.state_dict(), model_save_path)
                    #                     print(f"New best model saved at {model_save_path}")
                    best_iou = epoch_test_iou
                    model_save_path = os.path.join(exp_name, f'best_model_IoU{best_iou:.4f}.pth')  # Renamed for clarity

                    # Save a comprehensive checkpoint for resuming
                    checkpoint_state = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_iou': best_iou,
                        'run_losses': run_losses,
                        'test_losses': test_losses
                    }
                    torch.save(checkpoint_state, model_save_path)  # Save best model as a full checkpoint
                    print(f"New best model (checkpoint) saved at {model_save_path}")

            acc_metric_test.reset()
            ppv_metric_test.reset()
            tpr_metric_test.reset()
            iou_metric_test.reset()

        #         if (epoch + 1) in [50, 75, 100]:
        #             checkpoint_path = os.path.join(exp_name, f'trained_model_checkpoint_epoch{epoch+1}.pth')
        #             torch.save(model.state_dict(), checkpoint_path)
        #             print(f"Checkpoint model saved at {checkpoint_path}")
        if ((epoch + 1) % args.ckpt_interval == 0) or (epoch + 1 == args.epochs):  # Also save at final epoch
            checkpoint_path = os.path.join(exp_name, f'checkpoint_epoch_{epoch + 1}.pth')  # Renamed for clarity

            checkpoint_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,  # Best IoU up to this point
            }
            torch.save(checkpoint_state, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    print("\n========== TRAINING COMPLETE ==========")
    time_taken = datetime.now() - start_time
    print(f"Total training time: {time_taken}")

    np.save(os.path.join(exp_name, "run_losses.npy"), np.array(run_losses))
    np.save(os.path.join(exp_name, "test_losses.npy"), np.array(test_losses))
    np.save(os.path.join(exp_name, "run_accs.npy"), np.array(run_accs))
    np.save(os.path.join(exp_name, "test_accs.npy"), np.array(test_accs))
    np.save(os.path.join(exp_name, "run_TPRs.npy"), np.array(run_TPRs))
    np.save(os.path.join(exp_name, "test_TPRs.npy"), np.array(test_TPRs))
    np.save(os.path.join(exp_name, "run_PPVs.npy"), np.array(run_PPVs))
    np.save(os.path.join(exp_name, "test_PPVs.npy"), np.array(test_PPVs))
    np.save(os.path.join(exp_name, "run_IOUs.npy"), np.array(run_IOUs))
    np.save(os.path.join(exp_name, "test_IOUs.npy"), np.array(test_IOUs))
    print("Metrics saved.")

    # Close the SummaryWriter instance when training is complete
    writer.close()
    print("TensorBoard SummaryWriter closed.")


class Args:
    def __init__(self):
        self.exp_name = "mito_monai_unet_from_scratch_3"  # New experiment name for MONAI run
        self.train_zarr_dirs = ["/media/samia/DATA/mounts/fibserver1/smohinta_data/catena_data/HEMI_MITO/data_3d/train"]
        self.test_zarr_dirs = ["/media/samia/DATA/mounts/fibserver1/smohinta_data/catena_data/HEMI_MITO/data_3d/test"]
        self.label_type = 'neuron'
        self.epochs = 100
        self.batch_size = 1  # Keep at 1 for initial checks
        self.patch_size = [128, 128, 128]
        self.stride = [64, 64, 64]
        self.original_res = [8.0, 8.0, 8.0]
        self.target_res = [8.0, 8.0, 8.0]
        self.clahe = True
        self.subsample_frac = 0.2
        self.subsample_number = 0
        self.subsample_seed = 27
        self.balance_patches = True
        self.min_positive_pixels = 100
        self.learning_rate = 0.002
        self.learning_rate_after_hotstart_50 = 0.002
        self.loss_type = 'DiceCE'
        self.loss_weights = 5.0
        self.rotation_augs = True
        self.contrast_augs = True
        self.model_loc = None
        self.freeze_encoder = False
        self.hotstart = False
        self.save_images = True
        self.eval_interval = 5
        self.ckpt_interval = 25
        self.num_workers = 0
        self.resume_checkpoint = None # Path to a checkpoint file to resume training from


if __name__ == '__main__':
    args = Args()

    # Call the training function
    train(args)
