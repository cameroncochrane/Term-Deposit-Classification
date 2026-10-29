variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "eu-west-2"
}

variable "project_name" {
  description = "Name prefix for created resources"
  type        = string
  default     = "term-deposit-classification"
}

variable "ecr_repository_name" {
  description = "ECR repository holding the API container image"
  type        = string
  default     = "term-deposit-classification-api"
}

variable "image_tag" {
  # Defaults to whatever's currently live (the last successful CI deploy's commit SHA tag) so an
  # apply run without -var doesn't accidentally roll the deployment back to a stale image.
  description = "Tag of the image in ECR to deploy"
  type        = string
  default     = "8e34ab7d3857ddea10d8ab9e70db9d1857d2e654"
}

variable "lambda_memory_size" {
  # Lambda allocates CPU proportionally to memory. The cold-start import of pandas/scikit-learn/
  # xgboost was timing out at 1024MB (see CI incident: INIT_REPORT ... Status: timeout) - more
  # memory buys more CPU, which is what actually speeds up imports and unpickling the model.
  description = "Memory (MB) allocated to the Lambda function"
  type        = number
  default     = 2048
}

variable "lambda_timeout" {
  # API Gateway's own HTTP API integration timeout is hard-capped at 30s regardless of this
  # value, so this mostly matters for the CI warm-up invoke (a direct `aws lambda invoke`,
  # not through API Gateway) - it needs room for a worst-case cold start to actually finish.
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 60
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention period for the Lambda function"
  type        = number
  default     = 14
}

variable "github_repository" {
  description = "GitHub \"owner/repo\" allowed to assume the CI deploy role via OIDC"
  type        = string
  default     = "cameroncochrane/Term-Deposit-Classification"
}
