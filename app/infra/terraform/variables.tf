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
  description = "Tag of the image in ECR to deploy"
  type        = string
  default     = "manual-phase2-lambda"
}

variable "lambda_memory_size" {
  description = "Memory (MB) allocated to the Lambda function"
  type        = number
  default     = 1024
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 30
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
