# Food Segmentation

## Dataset

* [DataSet Images](https://mm.cs.uec.ac.jp/uecfoodpix/)
* [FoodSeg103-Benchmark-v1](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1)

## Related Links

* [Segment Anything](https://segment-anything.com/)
* [Segmentation Models](https://github.com/qubvel/segmentation_models)
* [Image Segmentation Keras](https://github.com/divamgupta/image-segmentation-keras)

## Experiments

| Dataset | Neural Network | Recall | Precision | F1 |
| :----: | :----: | :----: | :----: |
| food-seg-103 | [FPN SEResNet152](https://drive.google.com/file/d/1Gsaki177f22A-gGOR0Wkf-dEf214Fpkf/view?usp=share_link) | 63.31% | 45.45% | 52.91% |
| food-seg-103 | [U-Net DenseNet121](https://drive.google.com/file/d/1PwFIzXeCicoFoF8zddAYzn_6OMx23WV8/view?usp=share_link) | 54.95% | 42.64% | 48.02% |
| food-seg-103 | [LinkNet EfficientNetB3](https://drive.google.com/file/d/1HitaLjcA4o3OrjS87CuUChcidPv6fQVF/view?usp=drive_link) | 49.97% | 40.32% | 44.63% |

## Unresolved Bugs

* [Iou_score starts very high and keeps on decreasing/multi-class segmentation](https://github.com/qubvel/segmentation_models/issues/458)
