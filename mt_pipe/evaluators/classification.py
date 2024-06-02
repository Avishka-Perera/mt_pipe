import os
from os.path import join as ospj
from typing import Sequence, Dict
from PIL import Image
import numpy as np
import cv2 as cv

import torch

from ._base import Evaluator


class Classification(Evaluator):

    def __init__(
        self,
        top_k: Sequence[int],
        _result_dir: str,
        _norm_mean: Sequence[float] = [0, 0, 0],
        _norm_std: Sequence[float] = [1, 1, 1],
        export_size: int = 5,
        pad_frac: float = 0.02,
        id2cls: Dict[int, str] = None,
    ) -> None:
        self.top_k = top_k
        self.report_path = ospj(_result_dir, "report.txt")
        self.cls_imgs_dir = ospj(_result_dir, "classes")
        self.prd_imgs_dir = ospj(_result_dir, "predictions")
        os.makedirs(self.cls_imgs_dir, exist_ok=True)
        os.makedirs(self.prd_imgs_dir, exist_ok=True)
        self.correct = {k: 0 for k in top_k}
        self.total = 0
        self.id2cls = id2cls
        self.cls_imgs = {}
        self.ant_imgs = []
        self.n_exported_ant_imgs = 0
        self.norm_mean = torch.tensor(_norm_mean).reshape(1, 3, 1, 1)
        self.norm_std = torch.tensor(_norm_std).reshape(1, 3, 1, 1)
        self.export_size = export_size
        self.pad_frac = pad_frac

    def annotate_img(self, img: np.ndarray, lbl: int, prd: int) -> np.ndarray:
        h, w, _ = img.shape
        h_pad, w_pad = max(1, round(h * self.pad_frac)), max(
            1, round(w * self.pad_frac)
        )

        # color the borders
        color = [0, 255, 0] if lbl == prd else [255, 0, 0]
        img[:h_pad] = color
        img[:, :w_pad] = color
        img[-h_pad:] = color
        img[:, -w_pad:] = color

        # put text
        if self.id2cls is not None:
            prd, lbl = self.id2cls[prd], self.id2cls[lbl]
        img = img.astype(np.uint8).copy()
        text = f"predicted: {prd}, actual: {lbl}"
        org = [w_pad, h - h_pad]
        fontScale = w / 600
        thickness = max(1, round(w / 400 * 2))
        cv.putText(
            img,
            text,
            org,
            cv.FONT_HERSHEY_SIMPLEX,
            fontScale,
            [255, 255, 255],
            thickness,
        )

        return img

    def export_ant_imgs(self):
        if len(self.ant_imgs) > 0:
            h, w, _ = self.ant_imgs[0].shape
            export_img = np.zeros(
                [int(self.export_size * h), int(self.export_size * w), 3]
            ).astype(np.uint8)
            for i, img in enumerate(self.ant_imgs):
                col = i % self.export_size
                row = i // self.export_size
                export_img[
                    int(row * h) : int((row + 1) * h), int(col * w) : int((col + 1) * w)
                ] = img
            export_img = Image.fromarray(export_img)
            self.n_exported_ant_imgs += 1
            export_img.save(ospj(self.prd_imgs_dir, f"{self.n_exported_ant_imgs}.jpg"))

    def push_ant_img(self, img=None) -> None:
        if img is not None:
            if len(self.ant_imgs) < self.export_size**2:
                self.ant_imgs.append(img)
            else:
                self.export_ant_imgs()
                self.ant_imgs = [img]
        else:
            self.export_ant_imgs()
            self.ant_imgs = []

    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, images: torch.Tensor
    ) -> None:
        # accuracy metric calculations
        self.total += len(labels)
        for k in self.top_k:
            topk = logits.topk(k).indices
            self.correct[k] += (topk == labels.reshape(-1, 1)).any(dim=1).sum().item()

        # visualization caching
        images = images.cpu() * self.norm_std + self.norm_mean
        labels = labels.tolist()
        preds = topk[:, 0].tolist()
        for lbl, prd, img in zip(labels, preds, images):
            img = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # class images
            if lbl not in self.cls_imgs:
                self.cls_imgs[lbl] = img

            # annnotate image
            img = self.annotate_img(img, lbl, prd)
            self.push_ant_img(img)

    def export_result(self) -> str:
        # text report
        report = []
        for k, v in self.correct.items():
            report.append(f"Accuracy Top-{k}: {v/self.total*100}%")
        report = "\n".join(report)
        with open(self.report_path, "w") as handler:
            handler.write(report)

        # visualizations
        for lbl, img in self.cls_imgs.items():
            if self.id2cls is not None:
                lbl = self.id2cls[lbl]
            img_path = ospj(self.cls_imgs_dir, f"{lbl}.jpg")
            img = Image.fromarray(img)
            img.save(img_path)
        self.push_ant_img()  # export the residuals

        return report
