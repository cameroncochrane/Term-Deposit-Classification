# Remote state backend for Terraform itself, so CI (phase 5) and a human operator apply against
# the same state rather than CI silently diverging from the local state file. These two resources
# are created with the *local* backend still active - see infra/README.md for the one-time
# `terraform init -migrate-state` step that moves state into the bucket this creates.

resource "aws_s3_bucket" "tfstate" {
  bucket = "term-deposit-classification-tfstate-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# No DynamoDB lock table: Terraform's S3 backend now supports native state locking via
# conditional writes (`use_lockfile = true` in providers.tf), so a separate lock table isn't
# needed on current Terraform versions.
