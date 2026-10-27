# Live deployment stack

The self-contained production stack for the term deposit classifier. Nothing in here
depends on anything outside `app/` — Docker, ECR, and AWS only ever see this directory.

- `api/` — FastAPI app (`/health`, `/predict`), Pydantic schemas, the Lambda entrypoint. See
  [api/README.md](api/README.md).
- `models/` — the deployed `model_pipeline_2.joblib`.
- `Dockerfile` — builds the Lambda container image.
- `infra/terraform/` — provisions Lambda + API Gateway. See [infra/README.md](infra/README.md).
- `monitoring/` — Evidently drift monitoring (not yet implemented).
- `tests/` — API and Lambda-handler tests (`pytest app/tests/`).

Live endpoint: `https://aalr9ksiel.execute-api.eu-west-2.amazonaws.com`

## Updating the API

The Lambda function runs from a container image, so there's no "just push the code" option —
a code or model change means rebuild the image, push it to ECR, then point Lambda at the new one.

1. **Make the change** under `app/api/` (or replace `app/models/model_pipeline_2.joblib`).

2. **Test locally** before touching any image:

   ```bash
   cd app && uvicorn api.main:app --reload
   # in another shell:
   pytest app/tests/
   ```

3. **Rebuild the image.** Must be built for `linux/arm64` (matches the deployed
   `architectures`), and `--provenance=false --sbom=false` is required — Docker's default
   buildx attestations produce an image format Lambda's `CreateFunction`/`UpdateFunctionCode`
   rejects outright:

   ```bash
   docker build --platform linux/arm64 --provenance=false --sbom=false \
     -t term-deposit-api:<new-tag> -f app/Dockerfile app
   ```

4. **Push to ECR with a new tag.** The repo (`term-deposit-classification-api`) has
   `imageTagMutability: IMMUTABLE` — re-pushing an existing tag is rejected, so pick a new one
   each time (git short SHA is a reasonable default once this is automated in CI):

   ```bash
   ECR_URI="934912969749.dkr.ecr.eu-west-2.amazonaws.com/term-deposit-classification-api"
   aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin "${ECR_URI%%/*}"
   docker tag term-deposit-api:<new-tag> "${ECR_URI}:<new-tag>"
   docker push "${ECR_URI}:<new-tag>"
   ```

5. **Point Terraform at the new tag and apply:**

   ```bash
   cd app/infra/terraform
   terraform apply -var image_tag=<new-tag>
   ```

   Terraform resolves `<new-tag>` to its image digest and updates the Lambda function's
   `image_uri` — this is what actually redeploys the code. The next invocation cold-starts the
   new image.

6. **Smoke test the live endpoint:**

   ```bash
   API="https://aalr9ksiel.execute-api.eu-west-2.amazonaws.com"
   curl -s "$API/health"
   curl -s -X POST "$API/predict" -H "Content-Type: application/json" \
     -d '{"duration":120,"day":15,"month":"may","housing":"yes","job":"technician"}'
   ```

**Don't** run `aws lambda update-function-code` directly instead of step 5 — it works, but it
changes the real Lambda function without updating Terraform's state, so the next
`terraform plan` will see the old digest in `variables.tf` and try to revert the deployment back.
Once a resource is under Terraform, changes should go through it.
