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

## CI/CD

Two workflows, at the repo root in `.github/workflows/` (GitHub only ever reads workflows from
there, not from a nested `app/.github/` - a platform constraint, not a scope violation; every
step inside them still only touches `app/`):

- **[ci-cd.yml](../.github/workflows/ci-cd.yml)** - on every push: lint (`ruff`), test
  (`pytest app/tests/`), and a validation-only Docker build (not pushed). On push to `main`
  only, additionally: build and push the image to ECR tagged with the commit SHA, run
  `terraform apply -var image_tag=<sha>` to redeploy, then smoke-test `/health` and `/predict`
  against the real endpoint.
- **[rollback.yml](../.github/workflows/rollback.yml)** - manual (`workflow_dispatch`), takes a
  target image tag (a previous commit SHA) as input and runs the exact same
  `terraform apply -var image_tag=...` the deploy job uses - "redeploy the previous tag" rather
  than bespoke rollback logic. Available tags: check the `ci-cd.yml` run history, or
  `aws ecr list-images --repository-name term-deposit-classification-api`.

Both authenticate to AWS via GitHub's OIDC federation into a scoped IAM role
(`term-deposit-classification-github-actions-deploy`, see `infra/terraform/github_oidc.tf`) -
no long-lived AWS keys stored as GitHub secrets. The role can only push to this one ECR repo,
update this one Lambda function, and read/write this one Terraform state bucket - not broad
account access. It's scoped (via the OIDC trust condition) to `main` only, so PR/branch pushes
run lint/test/build with zero AWS access.

Terraform state lives in S3 (`term-deposit-classification-tfstate-<account-id>`, versioned,
native S3 locking) rather than locally, specifically so CI and a human operator apply against
the same state instead of silently diverging.

## Updating the API

**Pushing to `main` now does this automatically** (see CI/CD above). The manual steps below are
for local testing before you push, or an out-of-band deploy if you ever need one without CI.

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
   each time (CI tags by the full git commit SHA — `${{ github.sha }}`):

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
