from dataset import ConcatDataset, DataCollatorForSupervisedDataset, GroundingTSVDataset
from dataset.task_fns import GroundingTaskFn
from dataset.task_fns.task_prompts.grounding_task import (
    GROUNDING_SINGLE_REGION_STAGE_XYXY,
)

min_pixels = 16 * 28 * 28
max_pixels = 2560 * 28 * 28

model_name_or_path = "IDEA-Research/Rex-Omni"

grounding_data = dict(
    type=GroundingTSVDataset,
    img_tsv_file="Mountchicken/Rex-Omni-Finetune-ToyData/toy_data.images.tsv",
    ann_tsv_file="Mountchicken/Rex-Omni-Finetune-ToyData/toy_data.annotations.tsv",
    ann_lineidx_file="Mountchicken/Rex-Omni-Finetune-ToyData/toy_data.annotations.tsv.lineidx",
    image_min_pixels=min_pixels,
    image_max_pixels=max_pixels,
    task_fn=dict(
        type=GroundingTaskFn,
        task_prompts=GROUNDING_SINGLE_REGION_STAGE_XYXY,
        image_min_pixels=min_pixels,
        image_max_pixels=max_pixels,
    ),
    dataset_name="rexomni_grounding_data",
)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        grounding_data,
    ],
)

data_collator = dict(type=DataCollatorForSupervisedDataset)
