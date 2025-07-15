# submit_job.py
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import JobResourceConfiguration

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential,
    subscription_id="898ebc3c-57d6-4f37-b363-18025f16ef18",
    resource_group_name="CloudCityManagedServices",
    workspace_name="playground",
)

# ------------------------------------------------------------------
# INPUT: Train-Ordner aus workspaceblobstore → wird ins Job-WD geladen
# ------------------------------------------------------------------
train_data = Input(
    type="uri_folder",
    path=(
        "azureml://subscriptions/898ebc3c-57d6-4f37-b363-18025f16ef18/resourcegroups/CloudCityManagedServices/workspaces/playground/datastores/workspaceblobstore/paths/LocalUpload/b95a898dcf83a4411039e5e281a332e8591a17a107e49ee33a13b987c48e436d/splits/"
    ),
    mode="download",          # Dateien vor Job-Start kopieren
)

out_artifacts = Output(type=AssetTypes.URI_FOLDER, mode="upload")


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
USE_LOCAL_MODEL = False            # Variante B = True

if USE_LOCAL_MODEL:
    pretrained_model = Input(
        type = AssetTypes.URI_FOLDER,
        path = "azureml://datastores/workspaceblobstore/paths/models/svd_xt",
        mode = InputOutputModes.RO_MOUNT,
    )
    model_arg = "${{inputs.pretrained_model}}"
else:
    model_arg = "stabilityai/stable-video-diffusion-img2vid-xt"

# ------------------------------------------------------------------
# Neuen Job anlegen
# ------------------------------------------------------------------
job = command(
    code="./",                         # Repo-Root wird hochgeladen
    outputs={"artifacts": out_artifacts},
    environment="svd-h100-cu124:5",
    compute="NC96ads-A100-2",
    resources=JobResourceConfiguration(shm_size="128g"),
    experiment_name="dieter_SVD_Training_1e4_bs_1",
    display_name="train_SVD_lr_1e4_bs_1",

    # --------------------------------------------------------------
    # Achtung: langer Inline-Befehl!
    # --------------------------------------------------------------
    command = f"""
        pip install bitsandbytes && \
        pip install torchmetrics && \

        export DATA_ROOT_TESTVIDS=${{inputs.train_data}}/TestReference/test/videos && \
        export DATA_ROOT=${{inputs.train_data}}/train && \
        export DATA_ROOT_TEST=${{inputs.train_data}} && \
        export OUTPUT_DIR=./outputs && \
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128" && \

        accelerate launch --num_processes ${{{{inputs.num_processes}}}} --num_machines ${{{{inputs.num_machines}}}} --mixed_precision ${{{{inputs.mixed_precision}}}} \
            SVD_Xtend/train_svd_lora.py \
                --base_folder ${{{{inputs.train_data}}}}/train \
                --pretrained_model_name_or_path {model_arg} \
                --num_frames ${{{{inputs.num_frames}}}} \
                --width ${{{{inputs.width}}}} \
                --height ${{{{inputs.height}}}} \
                --output_dir "$AZUREML_OUTPUTS_PATH" \
                --per_gpu_batch_size ${{{{inputs.per_gpu_batch_size}}}} \
                --num_train_epochs ${{{{inputs.epochs}}}} \
                --gradient_accumulation_steps ${{{{inputs.gradient_accumulation}}}} \
                --learning_rate ${{{{inputs.learning_rate}}}} \
                --scale_lr \
                --lr_scheduler ${{{{inputs.lr_scheduler}}}} \
                --lr_warmup_steps ${{{{inputs.lr_warmup_steps}}}} \
                --num_workers ${{{{inputs.num_workers}}}} \
                --validation_steps ${{{{inputs.validation_steps}}}} \
                --num_validation_images ${{{{inputs.num_validation_images}}}} \
                --checkpointing_steps ${{{{inputs.checkpointing_steps}}}} \
                --checkpoints_total_limit ${{{{inputs.checkpoints_total_limit}}}} \
                --use_8bit_adam \
                --allow_tf32 \
                --gradient_checkpointing \
                --rank ${{{{inputs.rank}}}} \
                --conditioning_dropout_prob ${{{{inputs.conditioning_dropout_prob}}}} \
                --seed ${{{{inputs.seed}}}} \
                --report_to tensorboard && \

        python inference_batch_svd.py && \
        python metrics/compute_metrics.py \
            --mse true \
            --ssim true \
            --intrusion false \
            --gt_dir ${{DATA_ROOT_TESTVIDS}} \
            --gen_dir ${{OUTPUT_DIR}}/ValidationsLastCheckpoint \
            --out_file ${{OUTPUT_DIR}}/results.txt
    """,
    inputs = {
        "train_data": train_data,
        **({"pretrained_model": pretrained_model} if USE_LOCAL_MODEL else {}),
        "num_machines":             1,
        "num_processes":            4,
        "mixed_precision":          "bf16",
        "num_frames":               17,
        "width":                    640,
        "height":                   640,
        "per_gpu_batch_size":       1,
        "gradient_accumulation":    1,
        "epochs":                   5,
        "learning_rate":            1e-4,
        "lr_scheduler":             "cosine",
        "lr_warmup_steps":          50,
        "num_workers":              12,
        "validation_steps":         1000000,
        "num_validation_images":    1,
        "checkpointing_steps":      888,
        "checkpoints_total_limit":  20,
        "rank":                     8,
        "conditioning_dropout_prob":0,
        "seed":                     42,
    },
    environment_variables = {
        "BNB_CUDA_VERSION": "124",
        # "HF_TOKEN": "${{secrets.hf_token}}",
    },
)


print("Submitting job …")
run = ml_client.create_or_update(job)
print("Job submitted:", run.name) 