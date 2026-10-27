# syntax=docker/dockerfile:1
#
# Build with `docker build --provenance=false --sbom=false ...`. Buildx's default attestations
# turn the pushed image into an OCI image index (image manifest + attestation manifest), which
# Lambda's CreateFunction rejects with "image manifest ... is not supported."

FROM public.ecr.aws/lambda/python:3.12 AS builder

COPY requirements.txt .
# nvidia-nccl-cu12 is pulled in unconditionally by xgboost's Linux wheel (platform_system == "Linux")
# even though it's only needed for multi-GPU training; CPU inference never imports it, so drop it to
# avoid ~350MB of dead weight in the Lambda image.
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}" \
    && rm -rf "${LAMBDA_TASK_ROOT}"/nvidia "${LAMBDA_TASK_ROOT}"/nvidia_nccl_cu12*

FROM public.ecr.aws/lambda/python:3.12

COPY --from=builder ${LAMBDA_TASK_ROOT} ${LAMBDA_TASK_ROOT}

COPY api ${LAMBDA_TASK_ROOT}/api
COPY models ${LAMBDA_TASK_ROOT}/models

CMD ["api.lambda_handler.handler"]
