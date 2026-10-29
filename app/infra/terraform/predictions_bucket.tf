# Logged inputs of every live /predict call (app/api/prediction_log.py), compared against
# reference_data.csv by the drift monitor. Not versioned - these are disposable log objects, not
# state to recover.

resource "aws_s3_bucket" "predictions" {
  bucket = "${var.project_name}-predictions-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "predictions" {
  bucket = aws_s3_bucket.predictions.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "predictions" {
  bucket                  = aws_s3_bucket.predictions.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "predictions" {
  bucket = aws_s3_bucket.predictions.id

  rule {
    id     = "expire-old-predictions"
    status = "Enabled"

    filter {
      prefix = "predictions/"
    }

    expiration {
      days = 90
    }
  }
}
