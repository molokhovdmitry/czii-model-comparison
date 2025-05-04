import os
from typing import Optional, Any, Dict, List, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from matplotlib import pyplot as plt
from pytorch_toolbelt.optimization.functional import build_optimizer_param_groups
from pytorch_toolbelt.utils import all_gather, split_across_nodes, get_rank

from torch import Tensor
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from cryoet.modelling.detection.detection_head import ObjectDetectionOutput
from cryoet.schedulers import WarmupCosineScheduler
from .args import MyTrainingArguments, ModelArguments, DataArguments
from .od_accumulator import AccumulatedObjectDetectionPredictionContainer
from .visualization import render_heatmap
from ..data.parsers import CLASS_LABEL_TO_CLASS_NAME, ANGSTROMS_IN_PIXEL, TARGET_SIGMAS, TARGET_6_CLASSES, TARGET_5_CLASSES
from ..metric import score_submission
from ..modelling.detection.functional import decode_detections_with_nms


class ObjectDetectionModel(L.LightningModule):
    def __init__(
        self,
        *,
        model,
        data_args: DataArguments,
        model_args: ModelArguments,
        train_args: MyTrainingArguments,
    ):
        super().__init__()
        self.model = model
        self.data_args = data_args
        self.train_args = train_args
        self.model_args = model_args
        self.validation_predictions = None
        self.average_tokens_across_devices = train_args.average_tokens_across_devices
        self.num_classes = model.config.num_classes
        self.class_names = [cls["name"] for cls in (TARGET_6_CLASSES if self.num_classes == 6 else TARGET_5_CLASSES)]
        # fmt: off
        self.register_buffer("thresholds", torch.tensor(
            np.linspace(0.1, 0.9, num=161, dtype=np.float32)
        ))
        # fmt: on

        self.register_buffer("per_class_scores", torch.zeros(len(self.thresholds), self.num_classes))
        self.register_buffer("per_class_f1_scores", torch.zeros(len(self.thresholds), self.num_classes))
        self.inference_window_size = (
            model_args.valid_depth_window_size,
            model_args.valid_spatial_window_size,
            model_args.valid_spatial_window_size,
        )

    def forward(self, volume, labels=None, **loss_kwargs):
        return self.model(
            volume=volume,
            labels=labels,
            apply_loss_on_each_stride=self.model_args.apply_loss_on_each_stride,
            average_tokens_across_devices=self.average_tokens_across_devices,
            use_l1_loss=self.train_args.use_l1_loss,
            use_offset_head=self.model_args.use_offset_head,
            assigner_max_anchors_per_point=self.model_args.assigner_max_anchors_per_point,
            assigned_min_iou_for_anchor=self.model_args.assigned_min_iou_for_anchor,
            assigner_alpha=self.model_args.assigner_alpha,
            assigner_beta=self.model_args.assigner_beta,
            use_varifocal_loss=self.model_args.use_varifocal_loss,
            use_cross_entropy_loss=self.model_args.use_cross_entropy_loss,
            **loss_kwargs,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            **batch,
        )

        self.log_dict(
            dict(("train/" + k, v) for k, v in outputs.loss_dict.items()),
            batch_size=len(batch["volume"]),
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs.loss

    def on_train_epoch_start(self) -> None:
        torch.cuda.empty_cache()

    def on_validation_start(self) -> None:
        self.validation_predictions: Dict[str, AccumulatedObjectDetectionPredictionContainer] = {}
        for study in self.trainer.datamodule.valid_studies:
            self.validation_predictions[study] = AccumulatedObjectDetectionPredictionContainer.from_shape(
                shape=(184, 630, 630),  # Hard-coded so far, TODO pull from data module
                window_size=self.inference_window_size,
                num_classes=self.num_classes,
                strides=[2],  # Hard-coded so far, TODO pull from model
                device="cpu",
                dtype=torch.float16,
                use_weighted_average=self.data_args.use_weighted_average,
            )

    def accumulate_predictions(self, outputs: ObjectDetectionOutput, batch):
        tile_offsets_zyx = batch["tile_offsets_zyx"]

        scores = [torch.sigmoid(p).cpu() for p in outputs.logits]
        offsets = [p.cpu() for p in outputs.offsets]

        batch_size = len(batch["study"])
        for i in range(batch_size):
            study = batch["study"][i]
            volume_shape = batch["volume_shape"][i]
            tile_coord = tile_offsets_zyx[i]

            assert tuple(volume_shape) == (184, 630, 630), f"Volume shape is {volume_shape}"
            self.validation_predictions[study].accumulate(
                scores_list=[s[i] for s in scores],
                offsets_list=[o[i] for o in offsets],
                tile_coords_zyx=tile_coord,
            )

    def on_validation_epoch_end(self) -> None:
        torch.cuda.empty_cache()

        submission = dict(
            experiment=[],
            particle_type=[],
            score=[],
            x=[],
            y=[],
            z=[],
        )

        score_thresholds = self.thresholds.cpu().numpy()

        weights = {
            "apo-ferritin": 1,
            "beta-amylase": 0,
            "beta-galactosidase": 2,
            "ribosome": 1,
            "thyroglobulin": 2,
            "virus-like-particle": 1,
        }

        all_scores = {}
        all_offsets = {}
        all_studies = self.trainer.datamodule.valid_studies

        for study_name in all_studies:
            preds = self.validation_predictions.get(study_name, None)
            preds = all_gather(preds)

            preds = [p for p in preds if p is not None]
            if len(preds) == 0:
                continue

            accumulated_predictions = preds[0]
            for p in preds[1:]:
                accumulated_predictions += p

            scores, offsets = accumulated_predictions.merge_()
            all_scores[study_name] = scores
            all_offsets[study_name] = offsets

            # self.log_heatmaps(study_name, scores)

        # By this point, all_scores and all_offsets are gathered and we can distribute postprocessing across nodes
        rank_local_studies = split_across_nodes(all_studies)

        print(f"Rank {get_rank()} got local studies {rank_local_studies}")

        for study_name in rank_local_studies:
            # Save averaged heatmap for further postprocessing hyperparam tuning
            # if self.trainer.is_global_zero:
            #     torch.save({"scores": scores, "offsets": offsets}, os.path.join(self.trainer.log_dir, f"{study_name}.pth"))
            #
            #     self.trainer.datamodule.solution.to_csv(
            #         os.path.join(self.trainer.log_dir, f"{study_name}.csv"),
            #     )

            topk_coords_px, topk_clases, topk_scores = decode_detections_with_nms(
                all_scores[study_name],
                all_offsets[study_name],
                strides=[2],
                class_sigmas=TARGET_SIGMAS,
                min_score=score_thresholds.min(),
                iou_threshold=0.8,
                use_centernet_nms=self.model_args.use_centernet_nms,
                use_single_label_per_anchor=self.model_args.use_single_label_per_anchor,
                pre_nms_top_k=16536,
            )
            topk_scores = topk_scores.float().cpu().numpy()
            topk_coords = topk_coords_px.float().cpu().numpy() * ANGSTROMS_IN_PIXEL
            topk_clases = topk_clases.cpu().numpy()

            for cls, coord, score in zip(topk_clases, topk_coords, topk_scores):
                submission["experiment"].append(study_name)
                submission["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[int(cls)])
                submission["score"].append(float(score))
                submission["x"].append(float(coord[0]))
                submission["y"].append(float(coord[1]))
                submission["z"].append(float(coord[2]))

            # print("Added predictions for", study_name, "to dataframe")

        submission = pd.DataFrame.from_dict(submission)
        print(f"Rank {get_rank()} has {len(submission)} predictions")
        submission = all_gather(submission)
        submission = pd.concat(submission, ignore_index=True)
        print("Rank", get_rank(), "has", len(submission), "predictions after all_gather")

        # self.trainer.print(submission.sort_values(by="score", ascending=False).head(20))

        score_details = []
        f1_score_details = []
        for score_threshold in score_thresholds:
            keep_mask = submission["score"] >= score_threshold
            submission_filtered = submission[keep_mask]
            (f_beta, f1), particle_scores = score_submission(
                solution=self.trainer.datamodule.solution.copy(),
                submission=submission_filtered.copy(),
                row_id_column_name="id",
                distance_multiplier=0.5,
                beta=4,
            )
            # Extract F-beta and F1 scores for each particle type
            fbeta_scores = {k: v["f_beta"] for k, v in particle_scores.items()}
            f1_scores = {k: v["f1"] for k, v in particle_scores.items()}
            score_details.append(fbeta_scores)
            f1_score_details.append(f1_scores)

        keys = self.class_names
        # Process F-beta scores (competition metric)
        per_class_scores = []
        for scores_dict in score_details:
            per_class_scores.append([scores_dict[k] for k in keys])
        per_class_scores = np.array(per_class_scores)  # [threshold, class]

        # Process F1 scores
        per_class_f1_scores = []
        for scores_dict in f1_score_details:
            per_class_f1_scores.append([scores_dict[k] for k in keys])
        per_class_f1_scores = np.array(per_class_f1_scores)  # [threshold, class]

        # F-beta (competition metric)
        best_index_per_class = np.argmax(per_class_scores, axis=0)  # [class]
        best_threshold_per_class = np.array([score_thresholds[i] for i in best_index_per_class])  # [class]
        best_score_per_class = np.array([per_class_scores[i, j] for j, i in enumerate(best_index_per_class)])  # [class]
        averaged_score = np.sum([weights[k] * best_score_per_class[i] for i, k in enumerate(keys)]) / sum(weights.values())

        # F1 score
        best_f1_index_per_class = np.argmax(per_class_f1_scores, axis=0)  # [class]
        best_f1_threshold_per_class = np.array([score_thresholds[i] for i in best_f1_index_per_class])  # [class]
        best_f1_score_per_class = np.array([per_class_f1_scores[i, j] for j, i in enumerate(best_f1_index_per_class)])  # [class]
        averaged_f1_score = np.sum([weights[k] * best_f1_score_per_class[i] for i, k in enumerate(keys)]) / sum(weights.values())

        self.per_class_scores = torch.from_numpy(per_class_scores).to(self.device)
        self.per_class_f1_scores = torch.from_numpy(per_class_f1_scores).to(self.device)

        # Log F-beta score plot
        self.log_plots(
            dict((key, (score_thresholds, per_class_scores[:, i])) for i, key in enumerate(keys)), 
            "Threshold", "F-beta Score"
        )
        
        # Log F1 score plot
        self.log_plots(
            dict((key, (score_thresholds, per_class_f1_scores[:, i])) for i, key in enumerate(keys)), 
            "Threshold", "F1 Score",
            name_suffix="_f1"
        )

        # Log both F-beta and F1 metrics
        self.log_dict(
            {
                "val/score": averaged_score,  # Keep existing competition metric (F-beta) as the main score
                "val/f1_score": averaged_f1_score,  # Add F1 score
                **{f"val/{k}": best_score_per_class[i] for i, k in enumerate(keys)},
                **{f"val/{k}_f1": best_f1_score_per_class[i] for i, k in enumerate(keys)},
                **{f"val/{k}_threshold": best_threshold_per_class[i] for i, k in enumerate(keys)},
                **{f"val/{k}_f1_threshold": best_f1_threshold_per_class[i] for i, k in enumerate(keys)},
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            rank_zero_only=False,
        )

    def log_plots(self, plots: Dict[str, Tuple[np.ndarray, np.ndarray]], x_title, y_title, name_suffix=""):
        if self.trainer.is_global_zero:
            f = plt.figure()

            for key, (x, y) in plots.items():
                plt.plot(x, y, label=key)

            plt.xlabel(x_title)
            plt.ylabel(y_title)
            plt.legend()
            plt.tight_layout()

            tb_logger = self._get_tb_logger(self.trainer)
            if tb_logger is not None:
                tb_logger.add_figure(f"val/score_plot{name_suffix}", f, global_step=self.global_step)

            plt.close(f)

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)

        self.accumulate_predictions(outputs, batch)

        self.log_dict(
            dict(("val/" + k, v) for k, v in outputs.loss_dict.items()),
            batch_size=len(batch["volume"]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return outputs

    def _get_tb_logger(self, trainer) -> Optional[SummaryWriter]:
        tb_logger: Optional[TensorBoardLogger] = None
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger
                break

        if tb_logger is not None:
            return tb_logger.experiment
        return None

    def log_heatmaps(self, study_name: str, heatmaps: List[Tensor]):
        if self.trainer.is_global_zero:
            tb_logger = self._get_tb_logger(self.trainer)
            for i, heatmap in enumerate(heatmaps):
                heatmap = render_heatmap(heatmap)

                if tb_logger is not None:
                    tb_logger.add_images(
                        tag=f"val/{study_name}_{i}",
                        img_tensor=heatmap,
                        global_step=self.global_step,
                        dataformats="HWC",
                    )

    def configure_optimizers(self):
        param_groups, optimizer_kwargs = build_optimizer_param_groups(
            model=self,
            learning_rate=self.train_args.learning_rate,
            weight_decay=self.train_args.weight_decay,
            apply_weight_decay_on_bias=False,
            apply_weight_decay_on_norm=False,
        )

        if self.train_args.optim == "adamw_8bit":
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(param_groups, **optimizer_kwargs)
        elif self.train_args.optim == "paged_adamw_8bit":
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                betas=(self.train_args.adam_beta1, self.train_args.adam_beta2),
                **optimizer_kwargs,
                is_paged=True,
            )
        elif self.train_args.optim == "adamw_torch_fused":
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=(self.train_args.adam_beta1, self.train_args.adam_beta2),
                **optimizer_kwargs,
                fused=True,
            )
        elif self.train_args.optim == "adamw_torch":
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=(self.train_args.adam_beta1, self.train_args.adam_beta2),
                **optimizer_kwargs,
            )
        elif self.train_args.optim == "radam":
            optimizer = torch.optim.RAdam(
                param_groups,
                betas=(self.train_args.adam_beta1, self.train_args.adam_beta2),
                **optimizer_kwargs,
            )
        elif self.train_args.optim == "sgd":
            optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True, **optimizer_kwargs)
        else:
            raise KeyError(f"Unknown optimizer {self.train_args.optim}")

        warmup_steps = 0
        if self.train_args.warmup_steps > 0:
            warmup_steps = self.train_args.warmup_steps
            self.trainer.print(f"Using warmup steps: {warmup_steps}")
        elif self.train_args.warmup_ratio > 0:
            warmup_steps = int(self.train_args.warmup_ratio * self.trainer.estimated_stepping_batches)
            self.trainer.print(f"Using warmup steps: {warmup_steps}")

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            warmup_learning_rate=1e-7,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "cosine_with_warmup",
            },
        }

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]) -> None:
        scheduler.step()

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # Log gradient norm (mean and max)
        if self.global_step % self.train_args.logging_steps == 0:
            with torch.no_grad():
                all_grads = torch.stack(
                    [torch.norm(p.grad) for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                ).view(-1)

                grads_nan = [
                    n for n, p in self.model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()
                ]
                if len(grads_nan) > 0:
                    self.trainer.print(f"Found NaN gradients in {grads_nan}")

                max_grads = torch.max(all_grads).item()
                mean_grads = all_grads.mean()
                self.log(
                    "train/mean_grad",
                    mean_grads,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    "train/max_grad",
                    max_grads,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )
