output "api_endpoint" {
  description = "Base URL of the deployed API"
  value       = aws_apigatewayv2_api.api.api_endpoint
}

output "lambda_function_name" {
  description = "Name of the deployed Lambda function"
  value       = aws_lambda_function.api.function_name
}

output "github_actions_deploy_role_arn" {
  description = "Role ARN GitHub Actions assumes via OIDC to deploy - set as the AWS_DEPLOY_ROLE_ARN repo variable"
  value       = aws_iam_role.github_actions_deploy.arn
}

output "tfstate_bucket" {
  description = "S3 bucket holding Terraform state"
  value       = aws_s3_bucket.tfstate.bucket
}
