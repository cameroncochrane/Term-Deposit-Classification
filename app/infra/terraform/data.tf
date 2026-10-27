data "aws_caller_identity" "current" {}

data "aws_ecr_image" "api" {
  repository_name = var.ecr_repository_name
  image_tag       = var.image_tag
}
