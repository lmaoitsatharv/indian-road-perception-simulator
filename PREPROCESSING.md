# IDD preprocessing

The cleaned split has 5,507 training pairs, 687 validation pairs, and 689 test pairs.
The CSV at `dataset/splits/pair_splits.csv` controls which pairs are loaded.

`idd_data.py` prepares images and masks in memory when a batch is requested:

- Images remain RGB at 512 x 512, become float32, and are scaled to [-1, 1].
- Masks become float32 with shape [512, 512, 1]. Their labels remain exactly 0/1.
- Training shuffles pairs and may flip both image and mask horizontally together.
- Validation and test keep their order and receive no augmentation.
- Batch size is 4. The last smaller batch is kept. Seed is 42.
- No processed image copies or full-dataset memory cache are created.

Do not divide masks by 255. Do not normalize the images a second time inside a
future model: this loader already applies MobileNetV2 preprocessing.

## Run on this PC

From the project folder, run in PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' '.\check_preprocessing.py'
```

The check runs actual TensorFlow batches, compares validation/test pixels with
the files, checks paired flip alignment and reproducibility, and writes
`preprocessing/batch_check.json` plus `preprocessing/batch_preview.png`.

To use the loaders later:

```python
from idd_data import build_datasets

datasets, counts = build_datasets('dataset/splits', batch_size=4, seed=42)
train_ds = datasets['train']
val_ds = datasets['val']
test_ds = datasets['test']
```

The notebook `IDD_Preprocessing.ipynb` contains the same loader code and a guided
batch check. It is self-contained so it can also be used in Colab after copying
the complete `splits` folder and changing `SPLITS_DIR` in its settings cell.
The historical absolute paths inside the CSV are ignored; paths are reconstructed
from the current folder and numeric pair IDs.

## Dataset history

The 110 entirely zero-mask pairs were excluded from prepared split copies under the recorded filtering decision. Original source files and the removal record remain preserved. Do not
rerun `prepare_idd_splits.py` to prepare training data: that script documents the
initial split before the exclusions. Use the existing cleaned manifest.

Original driving-sequence IDs are unavailable, so the split cannot guarantee
that related driving frames are kept together. The preprocessing check does not
resolve that limitation or verify every mask's semantic correctness.

No U-Net is built and no training runs in this step.

## References

- [TensorFlow MobileNetV2 preprocessing](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/preprocess_input)
- [TensorFlow image segmentation tutorial](https://www.tensorflow.org/tutorials/images/segmentation)
