Terraform for the live stack: Lambda (container image) + API Gateway HTTP API.

```bash
cd app/infra/terraform
terraform init
terraform plan
terraform apply
```

Deploys the image already pushed to ECR (see `../requirements.txt` / `../Dockerfile`). Change
which image gets deployed with `-var image_tag=<tag>`. Default is `manual-phase2-lambda`.

State is local (`terraform.tfstate`, gitignored) for now — not shared, no locking. Fine for a
single operator; revisit (S3 backend + DynamoDB lock table) before more than one person applies
against this.
